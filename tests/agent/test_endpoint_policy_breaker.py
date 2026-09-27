"""Deterministic unit tests for endpoint_policy and endpoint_breaker (no network, no SDK)."""

import asyncio
import time
import pytest

from agent.endpoint_policy import (
    EndpointPolicy,
    EndpointKey,
    register_policy,
    get_policy,
    clear_registry as clear_policy_registry,
)
from agent.endpoint_breaker import (
    BreakerState,
    BreakerConfig,
    BreakerSnapshot,
    record_success,
    record_failure,
    can_attempt,
    attempt_slot,
    get_state,
    set_config,
    clear_registry as clear_breaker_registry,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    clear_policy_registry()
    clear_breaker_registry()
    yield
    clear_policy_registry()
    clear_breaker_registry()


# ============================================================================
# ENDPOINT POLICY TESTS
# ============================================================================

def test_default_policy_is_permissive():
    p = get_policy("any", "any", "any")
    assert isinstance(p, EndpointPolicy)
    assert p.ttft_timeout_s == 45.0
    assert p.chunk_idle_timeout_s == 60.0
    assert p.overall_deadline_s == 300.0


def test_register_and_get_policy():
    p = register_policy("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash",
                        ttft_timeout_s=10.0, chunk_idle_timeout_s=15.0, overall_deadline_s=60.0)
    assert p.ttft_timeout_s == 10.0
    assert p.chunk_idle_timeout_s == 15.0
    assert p.overall_deadline_s == 60.0

    got = get_policy("NVIDIA", "https://integrate.api.nvidia.com/v1/", "deepseek-v4.1-flash")
    # Same normalized key, values should match (not object identity due to frozen dataclass)
    assert got.ttft_timeout_s == 10.0
    assert got.chunk_idle_timeout_s == 15.0
    assert got.overall_deadline_s == 60.0


def test_policy_normalization():
    register_policy("NVIDIA", "https://EXAMPLE.COM/v1/", "model-x", ttft_timeout_s=5.0)
    assert get_policy("nvidia", "https://example.com/v1", "model-x").ttft_timeout_s == 5.0


def test_remaining_budget_basic():
    p = EndpointPolicy(ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=100.0)
    now = time.monotonic()
    deadline = now + 100.0
    assert p.remaining_budget(deadline, now) == 100.0
    assert p.remaining_budget(deadline, now + 30.0) == 70.0
    assert p.remaining_budget(deadline, now + 120.0) == 0.0


def test_per_attempt_ttft_bounded_by_budget():
    p = EndpointPolicy(ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=100.0)
    now = time.monotonic()
    deadline = now + 100.0
    assert p.per_attempt_ttft(deadline, now) == 45.0
    assert p.per_attempt_ttft(deadline, now + 30.0) == 45.0  # still > budget (70)
    assert p.per_attempt_ttft(deadline, now + 70.0) == 30.0  # bounded by remaining (30)
    assert p.per_attempt_ttft(deadline, now + 120.0) == 0.0


def test_read_timeout_for_sdk_not_undercutting_idle():
    p = EndpointPolicy(ttft_timeout_s=5.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    # read timeout MUST be >= chunk_idle_timeout_s
    assert p.read_timeout_for_sdk() >= p.chunk_idle_timeout_s
    # with default 45/60 -> read = 60+30 = 90
    p2 = EndpointPolicy(ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    assert p2.read_timeout_for_sdk() == 90.0


def test_policy_validation():
    with pytest.raises(ValueError):
        EndpointPolicy(ttft_timeout_s=0)
    with pytest.raises(ValueError):
        EndpointPolicy(chunk_idle_timeout_s=-1)
    with pytest.raises(ValueError):
        EndpointPolicy(overall_deadline_s=0)


# ============================================================================
# ENDPOINT BREAKER TESTS
# ============================================================================

def test_initial_state_closed():
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.CLOSED
    assert snap.half_open_probe_in_flight is False
    assert can_attempt("p", "u", "m") is True


def test_can_attempt_closed_always_true():
    for _ in range(10):
        assert can_attempt("p", "u", "m") is True


def test_qualifying_failure_in_closed_opens_breaker():
    record_failure("p", "u", "m", qualifies=True)
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.OPEN
    assert snap.open_until is not None
    assert snap.open_until > time.monotonic()


def test_non_qualifying_failure_in_closed_stays_closed():
    record_failure("p", "u", "m", qualifies=False)  # e.g. 400/401/403/404/422
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.CLOSED


def test_open_rejects_until_cooldown():
    record_failure("p", "u", "m", qualifies=True)
    # Immediately after -> OPEN, should reject
    assert can_attempt("p", "u", "m") is False


def test_open_transitions_to_half_open_after_cooldown():
    # Use very short cooldown for test
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    assert can_attempt("p", "u", "m") is False  # still OPEN
    time.sleep(0.02)
    # Cooldown expired -> should allow and transition to HALF_OPEN
    assert can_attempt("p", "u", "m") is True
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.HALF_OPEN
    assert snap.half_open_probe_in_flight is True


def test_half_open_allows_exactly_one_probe():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)
    # First caller gets the slot
    assert can_attempt("p", "u", "m") is True
    # Second caller should be rejected while probe in flight
    assert can_attempt("p", "u", "m") is False


def test_half_open_successful_probe_closes():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)
    can_attempt("p", "u", "m")  # enters HALF_OPEN
    record_success("p", "u", "m")
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.CLOSED
    assert snap.half_open_probe_in_flight is False


def test_half_open_qualifying_failure_reopens():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)
    can_attempt("p", "u", "m")  # enters HALF_OPEN
    record_failure("p", "u", "m", qualifies=True)  # probe fails
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.OPEN
    assert snap.open_until is not None


def test_half_open_non_qualifying_failure_closes():
    """Non-qualifying failure (config/client error) during probe -> CLOSED."""
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)
    can_attempt("p", "u", "m")  # enters HALF_OPEN
    record_failure("p", "u", "m", qualifies=False)  # 400/401/403/404/422 during probe
    snap = get_state("p", "u", "m")
    assert snap.state == BreakerState.CLOSED
    assert snap.half_open_probe_in_flight is False


def test_attempt_slot_context_manager():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)

    with attempt_slot("p", "u", "m"):
        snap = get_state("p", "u", "m")
        assert snap.state == BreakerState.HALF_OPEN
        assert snap.half_open_probe_in_flight is True
    # On exit, probe slot released
    snap = get_state("p", "u", "m")
    assert snap.half_open_probe_in_flight is False


def test_attempt_slot_releases_on_exception():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)

    try:
        with attempt_slot("p", "u", "m"):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    # Slot should be released even on exception
    snap = get_state("p", "u", "m")
    assert snap.half_open_probe_in_flight is False


def test_endpoint_isolation():
    """Different (provider, base_url, model) triples have independent breakers."""
    set_config("nvidia", "https://a.com/v1", "model-x", BreakerConfig(cooldown_s=0.01))
    set_config("nvidia", "https://b.com/v1", "model-x", BreakerConfig(cooldown_s=0.01))

    record_failure("nvidia", "https://a.com/v1", "model-x", qualifies=True)
    time.sleep(0.02)

    # First endpoint -> HALF_OPEN (after cooldown)
    assert can_attempt("nvidia", "https://a.com/v1", "model-x") is True

    # Second endpoint -> still CLOSED
    assert get_state("nvidia", "https://b.com/v1", "model-x").state == BreakerState.CLOSED
    assert can_attempt("nvidia", "https://b.com/v1", "model-x") is True


def test_breaker_snapshot_str():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=60.0))
    record_failure("p", "u", "m", qualifies=True)
    snap = get_state("p", "u", "m")
    s = str(snap)
    assert "OPEN" in s
    assert "cooldown" in s


def test_config_override():
    set_config("p", "u", "m", BreakerConfig(cooldown_s=1.0))
    entry = get_state("p", "u", "m")
    # Can't directly test internal config, but cooldown affects behavior
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(1.1)
    assert can_attempt("p", "u", "m") is True  # cooldown expired


# ============================================================================
# INTEGRATION-STYLE TESTS (policy + breaker together)
# ============================================================================

def test_policy_and_breaker_same_key():
    """Policy and breaker use same key normalization."""
    register_policy("NVIDIA", "https://API.EXAMPLE.COM/v1/", "MODEL-X", ttft_timeout_s=12.0)
    set_config("nvidia", "https://api.example.com/v1", "model-x", BreakerConfig(cooldown_s=1.0))

    p = get_policy("nvidia", "https://api.example.com/v1", "model-x")
    assert p.ttft_timeout_s == 12.0

    record_failure("NVIDIA", "https://API.EXAMPLE.COM/v1/", "MODEL-X", qualifies=True)
    snap = get_state("nvidia", "https://api.example.com/v1", "model-x")
    assert snap.state == BreakerState.OPEN


def test_concurrent_half_open_single_probe_threadsafe():
    """Only one thread may hold the HALF_OPEN probe under concurrency."""
    import threading as _th

    set_config("p", "u", "m", BreakerConfig(cooldown_s=0.01))
    record_failure("p", "u", "m", qualifies=True)
    time.sleep(0.02)

    wins = []
    barrier = _th.Barrier(10)

    def _try():
        barrier.wait()
        if can_attempt("p", "u", "m"):
            wins.append(1)

    threads = [_th.Thread(target=_try) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(wins) == 1, f"expected exactly 1 probe, got {len(wins)}"