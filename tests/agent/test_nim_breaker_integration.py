"""Point-3 verification: breaker wiring, probe invariant, cascade deadline, outcome mapping.

Deterministic — no network, no live provider calls. Exercises the REAL seams
(``endpoint_breaker``, ``_StreamingCall._record_breaker_outcome``, ``try_activate_fallback``).
"""

import threading
import time
from types import SimpleNamespace

import pytest

from agent import endpoint_breaker as eb
from agent import endpoint_policy as ep

NIM = ("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4.1-flash")
# A provider that has NOT opted in (no EndpointPolicy registered) — breaker must stay inert (§10).
OTHER = ("anthropic", "https://api.anthropic.com", "claude-sonnet-4.5")


@pytest.fixture(autouse=True)
def _isolate():
    eb.clear_registry()
    ep.clear_registry()
    # Opt-in: the breaker participates ONLY for endpoints with a registered EndpointPolicy (§10).
    ep.register_policy(*NIM)
    yield
    eb.clear_registry()
    ep.clear_registry()


def _trip(p, u, m):
    # Set the short cooldown BEFORE tripping: ``open_until`` is stamped at failure time from
    # the entry's config, so a later set_config() would not shorten the already-set window.
    eb.set_config(p, u, m, eb.BreakerConfig(cooldown_s=0.01))
    eb.record_failure(p, u, m, qualifies=True)


def _fast_cooldown(p, u, m):
    eb.set_config(p, u, m, eb.BreakerConfig(cooldown_s=0.01))


# ── §4 probe reservation invariant ──────────────────────────────────────────

def test_open_to_half_open_probe_release_then_reprobe():
    """The documented §4 sequence: OPEN -> cooldown -> HALF_OPEN -> probe -> cancel -> release
    -> the NEXT caller can probe again (no probe stuck reserved)."""
    p, u, m = NIM
    _trip(p, u, m)
    assert eb.get_state(p, u, m).state is eb.BreakerState.OPEN

    _fast_cooldown(p, u, m)
    time.sleep(0.02)
    assert eb.can_attempt(p, u, m) is True           # OPEN -> HALF_OPEN, probe reserved
    assert eb.get_state(p, u, m).half_open_probe_in_flight is True
    assert eb.can_attempt(p, u, m) is False          # single probe: second caller refused

    eb.release_probe(p, u, m)                        # caller cancellation
    snap = eb.get_state(p, u, m)
    assert snap.state is eb.BreakerState.HALF_OPEN   # released, NOT tripped
    assert snap.half_open_probe_in_flight is False
    assert eb.can_attempt(p, u, m) is True           # next caller may probe


def test_successful_probe_closes():
    p, u, m = NIM
    _trip(p, u, m)
    _fast_cooldown(p, u, m)
    time.sleep(0.02)
    assert eb.can_attempt(p, u, m) is True
    eb.record_success(p, u, m)
    assert eb.get_state(p, u, m).state is eb.BreakerState.CLOSED


def test_qualifying_probe_failure_reopens():
    p, u, m = NIM
    _trip(p, u, m)
    _fast_cooldown(p, u, m)
    time.sleep(0.02)
    assert eb.can_attempt(p, u, m) is True
    eb.record_failure(p, u, m, qualifies=True)
    assert eb.get_state(p, u, m).state is eb.BreakerState.OPEN


def test_nonqualifying_probe_failure_does_not_stick_half_open():
    """§8 edge case: a probe failing for a non-trip reason must not linger in HALF_OPEN."""
    p, u, m = NIM
    _trip(p, u, m)
    _fast_cooldown(p, u, m)
    time.sleep(0.02)
    assert eb.can_attempt(p, u, m) is True
    eb.record_failure(p, u, m, qualifies=False)
    assert eb.get_state(p, u, m).state is eb.BreakerState.CLOSED


def test_endpoint_isolation_same_model_different_provider():
    """(providerA, modelX) must not share breaker state with (providerB, modelX)."""
    _trip("providerA", "https://a/v1", "modelX")
    assert eb.get_state("providerA", "https://a/v1", "modelX").state is eb.BreakerState.OPEN
    assert eb.get_state("providerB", "https://b/v1", "modelX").state is eb.BreakerState.CLOSED


def test_concurrent_probe_admission_is_exactly_one():
    """§9 (class level): 10 concurrent acquisitions admit exactly one HALF_OPEN probe."""
    p, u, m = NIM
    _trip(p, u, m)
    _fast_cooldown(p, u, m)
    time.sleep(0.02)

    wins, barrier = [], threading.Barrier(10)

    def _go():
        barrier.wait()
        if eb.can_attempt(p, u, m):
            wins.append(1)

    ts = [threading.Thread(target=_go) for _ in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(wins) == 1, f"exactly one probe admitted, got {len(wins)}"


# ── §3 outcome mapping through the REAL streaming seam ─────────────────────

def _bare_streaming_call():
    from agent.chat_completion_helpers import _StreamingCall

    class _Self:
        pass

    s = _Self()
    s.agent = SimpleNamespace(provider=NIM[0], base_url=NIM[1], model=NIM[2])
    return _StreamingCall, s


def test_outcome_qualifying_failure_trips_breaker():
    _StreamingCall, s = _bare_streaming_call()
    _StreamingCall._record_breaker_outcome(s, qualifies=True)
    assert eb.get_state(*NIM).state is eb.BreakerState.OPEN


def test_outcome_success_closes_breaker():
    _StreamingCall, s = _bare_streaming_call()
    _trip(*NIM)
    _fast_cooldown(*NIM)
    time.sleep(0.02)
    assert eb.can_attempt(*NIM) is True
    _StreamingCall._record_breaker_outcome(s, success=True)
    assert eb.get_state(*NIM).state is eb.BreakerState.CLOSED


def test_outcome_cancellation_releases_without_tripping():
    _StreamingCall, s = _bare_streaming_call()
    _trip(*NIM)
    _fast_cooldown(*NIM)
    time.sleep(0.02)
    assert eb.can_attempt(*NIM) is True
    _StreamingCall._record_breaker_outcome(s, cancelled=True)
    snap = eb.get_state(*NIM)
    assert snap.state is eb.BreakerState.HALF_OPEN   # not falsely OPENed
    assert snap.half_open_probe_in_flight is False   # reservation released


def test_outcome_nonqualifying_is_noop_on_closed():
    _StreamingCall, s = _bare_streaming_call()
    _StreamingCall._record_breaker_outcome(s)  # neither success nor qualifies
    assert eb.get_state(*NIM).state is eb.BreakerState.CLOSED


# ── §5 cascade deadline ────────────────────────────────────────────────────

def _fake_agent(chain):
    a = SimpleNamespace(
        _fallback_chain=chain, _fallback_index=0, _cascade_deadline=None,
        _unavailable_fallback_keys=set(), _rate_limited_until=0.0,
        provider="nvidia", base_url="https://integrate.api.nvidia.com/v1", model="primary",
    )
    return a


def test_cascade_deadline_is_initialised_once(monkeypatch):
    import agent.chat_completion_helpers as h
    from agent import fallback_cooldown as fc

    monkeypatch.setattr(fc, "switch_deferred_by_reset", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_arm_rate_limit_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(h, "_fallback_chain_exhausted", lambda agent, reason: "EXHAUSTED")

    a = _fake_agent([])  # empty chain: returns exhausted before dialling anything
    h.try_activate_fallback(a, reason=None)
    assert a._cascade_deadline is not None, "first fallback attempt must start the clock"


def test_expired_cascade_deadline_stops_the_cascade(monkeypatch):
    import agent.chat_completion_helpers as h
    from agent import fallback_cooldown as fc

    monkeypatch.setattr(fc, "switch_deferred_by_reset", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_arm_rate_limit_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(h, "_fallback_chain_exhausted", lambda agent, reason: "EXHAUSTED")

    a = _fake_agent([{"provider": "nvidia", "model": "m", "base_url": "https://integrate.api.nvidia.com/v1"}])
    a._cascade_deadline = time.monotonic() - 1.0  # already spent
    assert h.try_activate_fallback(a, reason=None) == "EXHAUSTED"
    assert a._fallback_index == 0, "expired budget must not dial any endpoint"


def test_T17_global_deadline_expiry_does_not_trip_healthy_endpoint(monkeypatch):
    """Budget exhaustion != endpoint health failure: the breaker must stay CLOSED."""
    import agent.chat_completion_helpers as h
    from agent import fallback_cooldown as fc

    monkeypatch.setattr(fc, "switch_deferred_by_reset", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_arm_rate_limit_cooldown", lambda *a, **k: None)
    monkeypatch.setattr(h, "_fallback_chain_exhausted", lambda agent, reason: "EXHAUSTED")

    a = _fake_agent([{"provider": "nvidia", "model": "m", "base_url": "https://integrate.api.nvidia.com/v1"}])
    a._cascade_deadline = time.monotonic() - 1.0
    h.try_activate_fallback(a, reason=None)
    assert eb.get_state(*NIM).state is eb.BreakerState.CLOSED, "budget expiry must not open the endpoint"


def test_T19_one_attempt_yields_exactly_one_resolution(monkeypatch):
    """§2: one acquired attempt -> exactly ONE terminal breaker outcome (no double accounting)."""
    calls = []
    monkeypatch.setattr(eb, "record_success", lambda *a, **k: calls.append("success"))
    monkeypatch.setattr(eb, "record_failure", lambda *a, **k: calls.append("failure"))
    monkeypatch.setattr(eb, "release_probe", lambda *a, **k: calls.append("release"))

    _StreamingCall, s = _bare_streaming_call()
    _StreamingCall._record_breaker_outcome(s, qualifies=True)
    assert calls == ["failure"], f"exactly one resolution, got {calls}"

    calls.clear()
    _StreamingCall._record_breaker_outcome(s, success=True)
    assert calls == ["success"], f"exactly one resolution, got {calls}"

    calls.clear()
    _StreamingCall._record_breaker_outcome(s, cancelled=True)
    assert calls == ["release"], f"exactly one resolution, got {calls}"


def test_T20_breaker_is_opt_in_only():
    """§10: a non-opt-in endpoint gets NO breaker behaviour."""
    from agent.chat_completion_helpers import _StreamingCall

    class _Self:
        pass

    s = _Self()
    s.agent = SimpleNamespace(provider=OTHER[0], base_url=OTHER[1], model=OTHER[2])
    _StreamingCall._record_breaker_outcome(s, qualifies=True)   # would OPEN an opted-in endpoint
    assert eb.get_state(*OTHER).state is eb.BreakerState.CLOSED
    assert not ep.has_policy(*OTHER), "test premise: OTHER never opted in"
