"""Production-path proof — the NIM endpoint policy is ACTIVE in the real runtime.

The audit found `register_policy()` had no production call-site, so the policy,
the phase-aware watchdog and the breaker were NO-OP in the live runtime. These
tests exercise the REAL production functions (not `register_policy()` directly):

    _StreamingCall._resolve_stale_timeout          (real registration call-site)
    StreamingWaitMonitor._effective_stale_timeout  (real watchdog consumer)
    _StreamingCall._record_breaker_outcome         (real breaker outcome owner)

and prove that a NIM endpoint resolved by production code becomes policy-active,
while a non-opt-in provider keeps `has_policy == False` and its old behaviour.

Deterministic, no network.
"""

from types import SimpleNamespace

import pytest

from agent.endpoint_policy import (
    clear_registry,
    ensure_policy_for_endpoint,
    has_policy,
    is_nim_endpoint,
)
from agent.endpoint_breaker import can_attempt, clear_registry as _clear_breaker
from agent.chat_completion_helpers import _StreamingCall
from agent.chat_completion_stream_monitor import StreamingWaitMonitor

NIM = "https://integrate.api.nvidia.com/v1"

# The four NIM models planned for the main chain — shared provider + shared base_url,
# distinct model_id. The router identity must keep them separate.
NIM_CHAIN = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "z-ai/glm-5.3",
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
]


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    _clear_breaker()
    yield
    clear_registry()
    _clear_breaker()


def _host(provider: str, base_url: str, model: str) -> SimpleNamespace:
    """Minimal host exposing the attributes the REAL production methods read."""
    return SimpleNamespace(
        agent=SimpleNamespace(provider=provider, base_url=base_url, model=model),
        api_kwargs={"model": model},
        clients=SimpleNamespace(diag={}),
    )


# ── 1. Real registration call-site (primary path) ──────────────────────────────

def test_production_resolve_stale_timeout_registers_nim_policy():
    """Real `_resolve_stale_timeout` on a NIM identity -> policy ACTIVE + visible."""
    model = NIM_CHAIN[0]
    host = _host("nvidia", NIM, model)
    assert has_policy("nvidia", NIM, model) is False  # before: NO-OP

    _StreamingCall._resolve_stale_timeout(host)  # real production function body

    assert has_policy("nvidia", NIM, model) is True, "production path must register the policy"
    assert host._endpoint_policy is not None, "watchdog must receive the policy object"


def test_production_resolve_stale_timeout_leaves_non_nim_untouched():
    """Non-opt-in provider -> has_policy False, `_endpoint_policy` stays None."""
    host = _host("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-4")
    _StreamingCall._resolve_stale_timeout(host)
    assert has_policy("anthropic", "https://api.anthropic.com/v1", "claude-sonnet-4") is False
    assert host._endpoint_policy is None


# ── 2. Real watchdog consumer sees the policy ─────────────────────────────────

def test_production_watchdog_sees_phase_aware_policy_after_resolution():
    """`_effective_stale_timeout` returns the policy threshold, not the base patience."""
    model = NIM_CHAIN[0]
    host = _host("nvidia", NIM, model)
    # Pre-existing per-provider patience is the 180s cloud default; without a policy the
    # watchdog would keep it. With the policy active it must tighten to the TTFT bound.
    host._stream_stale_timeout = 180.0
    host._endpoint_policy = None
    assert StreamingWaitMonitor._effective_stale_timeout(host) == 180.0

    _StreamingCall._resolve_stale_timeout(host)  # real registration
    effective = StreamingWaitMonitor._effective_stale_timeout(host)
    assert effective < 180.0, "policy must be reachable through the real watchdog"


# ── 3. Real breaker outcome owner becomes reachable ───────────────────────────

def test_production_breaker_outcome_trips_only_after_registration():
    """`_record_breaker_outcome` is a no-op before wiring and trips after it."""
    model = NIM_CHAIN[0]
    host = _host("nvidia", NIM, model)

    # Before registration: non-opt-in -> no breaker behaviour at all.
    _StreamingCall._record_breaker_outcome(host, qualifies=True)
    assert can_attempt("nvidia", NIM, model) is True

    _StreamingCall._resolve_stale_timeout(host)  # real registration
    _StreamingCall._record_breaker_outcome(host, qualifies=True)
    assert can_attempt("nvidia", NIM, model) is False, "breaker must trip after wiring"


# ── 4. Identity distinctness — no provider-level collapse ─────────────────────

def test_nim_chain_keeps_four_distinct_endpoint_identities():
    """All four NIM models register independently; one trip does not affect the rest."""
    for model in NIM_CHAIN:
        host = _host("nvidia", NIM, model)
        _StreamingCall._resolve_stale_timeout(host)
    for model in NIM_CHAIN:
        assert has_policy("nvidia", NIM, model) is True

    # Trip ONE endpoint through the real outcome owner.
    _StreamingCall._record_breaker_outcome(_host("nvidia", NIM, NIM_CHAIN[1]), qualifies=True)
    assert can_attempt("nvidia", NIM, NIM_CHAIN[1]) is False, "tripped endpoint is closed"
    for model in (NIM_CHAIN[0], NIM_CHAIN[2], NIM_CHAIN[3]):
        assert can_attempt("nvidia", NIM, model) is True, f"{model} must be unaffected"


# ── 5. Opt-in scope + idempotency ─────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://integrate.api.nvidia.com/v1", True),
    ("https://integrate.api.nvidia.com/v1/", True),
    ("https://integrate.api.nvidia.com", True),
    ("https://api.anthropic.com/v1", False),
    ("https://generativelanguage.googleapis.com", False),
    ("https://bedrock-runtime.us-east-1.amazonaws.com", False),
    ("", False),
])
def test_opt_in_scope_is_nim_gateway_only(url, expected):
    assert is_nim_endpoint(url) is expected


def test_ensure_policy_is_idempotent_and_preserves_tuned_values():
    """A second call must not overwrite an already-tuned policy."""
    from agent.endpoint_policy import get_policy, register_policy
    assert ensure_policy_for_endpoint("nvidia", NIM, "m") is True
    register_policy("nvidia", NIM, "m", ttft_timeout_s=12.5, chunk_idle_timeout_s=99.0)
    assert ensure_policy_for_endpoint("nvidia", NIM, "m") is True
    tuned = get_policy("nvidia", NIM, "m")
    assert tuned.ttft_timeout_s == 12.5 and tuned.chunk_idle_timeout_s == 99.0
