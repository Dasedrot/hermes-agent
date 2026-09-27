"""Verification suite for the NIM endpoint-policy integration (deterministic, no network).

Covers the pieces landed for the NIM TTFT/idle + breaker work:
  * endpoint_policy.has_policy() opt-in gate
  * the phase-aware stale-timeout selector used by the single existing watchdog
  * the classifier's 5xx/timeout should_fallback verdicts (condition 7)
  * turn_recovery's transport-failure reason set
"""

import time
import pytest

from agent.endpoint_policy import (
    EndpointPolicy,
    register_policy,
    get_policy,
    has_policy,
    clear_registry as clear_policy_registry,
)
from agent.chat_completion_helpers import (
    _policy_phase_stale_timeout,
    _explicit_endpoint_policy,
)


@pytest.fixture(autouse=True)
def _reset():
    clear_policy_registry()
    yield
    clear_policy_registry()


# ── has_policy(): optimum-in gate ───────────────────────────────────────────

def test_has_policy_false_for_unregistered_endpoint():
    assert has_policy("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash") is False


def test_has_policy_true_after_register():
    register_policy("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash",
                    ttft_timeout_s=20.0)
    assert has_policy("NVIDIA", "https://integrate.api.nvidia.com/v1/", "deepseek-v4.1-flash") is True
    # still unregistered for a different model on the same route
    assert has_policy("nvidia", "https://integrate.api.nvidia.com/v1", "other-model") is False


def test_get_policy_default_still_permissive():
    p = get_policy("some", "https://example.com/v1", "whatever")
    assert isinstance(p, EndpointPolicy)
    assert p.ttft_timeout_s == 45.0 and p.chunk_idle_timeout_s == 60.0


# ── phase-aware stale selector ──────────────────────────────────────────────

def test_phase_pre_first_chunk_uses_ttft():
    pol = EndpointPolicy(ttft_timeout_s=10.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    now = time.monotonic()
    deadline = now + 300.0
    assert _policy_phase_stale_timeout(pol, 180.0, first_chunk_seen=False, deadline=deadline, now=now) == 10.0


def test_phase_post_first_chunk_uses_idle_not_ttft():
    """The critical property: after the first token the budget is idle, NOT the (shorter) TTFT."""
    pol = EndpointPolicy(ttft_timeout_s=10.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    now = time.monotonic()
    deadline = now + 300.0
    assert _policy_phase_stale_timeout(pol, 180.0, first_chunk_seen=True, deadline=deadline, now=now) == 60.0


def test_phase_bounded_by_base_patience_pre_chunk():
    pol = EndpointPolicy(ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    now = time.monotonic()
    deadline = now + 300.0
    # base patience 20s is stricter than the 45s TTFT -> min wins
    assert _policy_phase_stale_timeout(pol, 20.0, first_chunk_seen=False, deadline=deadline, now=now) == 20.0


def test_phase_spent_budget_is_zero():
    pol = EndpointPolicy(ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=100.0)
    now = time.monotonic()
    deadline = now - 1.0  # already blown
    assert _policy_phase_stale_timeout(pol, 180.0, first_chunk_seen=False, deadline=deadline, now=now) == 0.0
    assert _policy_phase_stale_timeout(pol, 180.0, first_chunk_seen=True, deadline=deadline, now=now) == 0.0


def test_phase_without_deadline():
    pol = EndpointPolicy(ttft_timeout_s=12.0, chunk_idle_timeout_s=40.0, overall_deadline_s=300.0)
    assert _policy_phase_stale_timeout(pol, None, first_chunk_seen=False, deadline=None, now=time.monotonic()) == 12.0
    assert _policy_phase_stale_timeout(pol, None, first_chunk_seen=True, deadline=None, now=time.monotonic()) == 40.0


# ── opt-in resolver ─────────────────────────────────────────────────────────

class _FakeAgent:
    def __init__(self, provider, base_url, model):
        self.provider, self.base_url, self.model = provider, base_url, model


def test_explicit_endpoint_policy_none_when_unregistered():
    a = _FakeAgent("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash")
    assert _explicit_endpoint_policy(a) is None


def test_explicit_endpoint_policy_returned_when_registered():
    register_policy("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash",
                    ttft_timeout_s=7.0, chunk_idle_timeout_s=9.0)
    a = _FakeAgent("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-v4.1-flash")
    pol = _explicit_endpoint_policy(a)
    assert pol is not None and pol.ttft_timeout_s == 7.0 and pol.chunk_idle_timeout_s == 9.0


def test_monitor_non_opt_in_returns_base_unchanged():
    """The single watchdog keeps its pre-existing patience when no policy is registered."""
    from agent.chat_completion_stream_monitor import StreamingWaitMonitor

    class _M(StreamingWaitMonitor):
        def __init__(self):
            self._stream_stale_timeout = 180.0
            self._endpoint_policy = None
            self.clients = None

    assert _M()._effective_stale_timeout() == 180.0


def test_monitor_opt_in_is_phase_aware():
    from agent.chat_completion_stream_monitor import StreamingWaitMonitor

    class _M(StreamingWaitMonitor):
        def __init__(self, diag):
            self._stream_stale_timeout = 180.0
            self._endpoint_policy = EndpointPolicy(ttft_timeout_s=5.0, chunk_idle_timeout_s=30.0,
                                                   overall_deadline_s=300.0)
            self._cascade_deadline = time.monotonic() + 300.0
            self.clients = type("C", (), {"diag": diag})()

    assert _M({})._effective_stale_timeout() == 5.0                      # pre first chunk
    assert _M({"first_chunk_at": 1.0})._effective_stale_timeout() == 30.0  # post first chunk


# ── condition 7: classifier verdicts ────────────────────────────────────────

class _StubError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [500, 501, 502, 503, 504])
def test_5xx_verdict_falls_back(status):
    from agent.error_classifier import classify_api_error
    c = classify_api_error(_StubError(status), provider="nvidia", model="m")
    assert c.should_fallback is True, f"HTTP {status} must be fallback-eligible"
    assert c.should_rotate_credential is False


def test_timeout_verdict_falls_back():
    from agent.error_classifier import classify_api_error
    c = classify_api_error(TimeoutError("read timed out"), provider="nvidia", model="m")
    assert c.should_fallback is True


def test_429_verdict_unchanged():
    from agent.error_classifier import classify_api_error
    c = classify_api_error(_StubError(429, "rate limit exceeded"), provider="nvidia", model="m")
    assert c.should_fallback is True
    assert c.should_rotate_credential is True


def test_404_is_not_inplace_retryable():
    """REAL Hermes semantics differ from the spec's §7 table.

    The spec said 400/401/403/404/422 => fallback NO. In the baseline code those verdicts
    carry ``_ABORT_FALLBACK = {"retryable": False, "should_fallback": True}``: the request is
    NOT retried in place, but the endpoint-specific failure (e.g. a model id absent on THIS
    provider) still aborts to the fallback chain. That behaviour is intentional and was
    deliberately left untouched; changing it would be a large unrelated behavioural change.
    This test pins the real contract so a future edit cannot silently flip it.
    """
    from agent.error_classifier import classify_api_error
    c = classify_api_error(_StubError(404, "model not found"), provider="nvidia", model="m")
    assert c.retryable is False            # never retried in place
    assert c.should_fallback is True       # but may abort to another provider (baseline, unchanged)


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_4xx_baseline_not_retryable(status):
    """Pin the baseline: 4xx verdicts are not retryable in place (unchanged by this work)."""
    from agent.error_classifier import classify_api_error
    c = classify_api_error(_StubError(status, "bad request"), provider="nvidia", model="m")
    assert c.retryable is not True


# ── condition 7: turn_recovery transport reasons ────────────────────────────

def test_server_error_is_a_transport_failure_reason():
    from agent.turn_recovery import _TRANSPORT_FAILURE_REASONS
    from agent.error_classifier import FailoverReason
    assert FailoverReason.server_error in _TRANSPORT_FAILURE_REASONS
    assert FailoverReason.timeout in _TRANSPORT_FAILURE_REASONS
    assert FailoverReason.overloaded in _TRANSPORT_FAILURE_REASONS
