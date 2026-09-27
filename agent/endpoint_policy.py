"""Per-endpoint streaming timeout policy (pure data, no IO).

An endpoint is identified by a stable triple:
    (provider, base_url, model_id)

Policy fields:
- ttft_timeout_s:   deadline until FIRST MEANINGFUL SSE event (content/reasoning/tool_calls)
- chunk_idle_timeout_s: deadline between subsequent meaningful SSE events
- overall_deadline_s:   cascade budget before first token; after first token only idle applies

Default policy is permissive (existing Hermes behaviour) — NIM endpoints opt in
via profile config or explicit registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import threading


EndpointKey = Tuple[str, str, str]  # (provider, base_url, model_id)


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """Timeout policy for a single endpoint."""

    ttft_timeout_s: float = 45.0
    chunk_idle_timeout_s: float = 60.0
    overall_deadline_s: float = 300.0  # cascade budget before first token

    def __post_init__(self) -> None:
        if self.ttft_timeout_s <= 0:
            raise ValueError("ttft_timeout_s must be > 0")
        if self.chunk_idle_timeout_s <= 0:
            raise ValueError("chunk_idle_timeout_s must be > 0")
        if self.overall_deadline_s <= 0:
            raise ValueError("overall_deadline_s must be > 0")

    def remaining_budget(self, deadline: float, now: float) -> float:
        """Remaining cascade budget before first token."""
        return max(0.0, deadline - now)

    def per_attempt_ttft(self, deadline: float, now: float) -> float:
        """Effective TTFT for this attempt, bounded by remaining cascade budget."""
        return min(self.ttft_timeout_s, self.remaining_budget(deadline, now))

    def read_timeout_for_sdk(self) -> float:
        """HTTPX read timeout to pass to SDK client — MUST be >= chunk_idle_timeout_s."""
        # SDK read timeout applies to entire stream; must not undercut idle policy.
        return max(self.chunk_idle_timeout_s, self.ttft_timeout_s) + 30.0


# Global registry — thread-safe, process-local
_registry: Dict[EndpointKey, EndpointPolicy] = {}
_registry_lock = threading.RLock()

# Default policy applied when no explicit registration exists
_DEFAULT_POLICY = EndpointPolicy()


def register_policy(
    provider: str,
    base_url: str,
    model_id: str,
    *,
    ttft_timeout_s: float = 45.0,
    chunk_idle_timeout_s: float = 60.0,
    overall_deadline_s: float = 300.0,
) -> EndpointPolicy:
    """Register or override policy for an endpoint."""
    key = (_norm(provider), _norm(base_url), _norm(model_id))
    policy = EndpointPolicy(
        ttft_timeout_s=ttft_timeout_s,
        chunk_idle_timeout_s=chunk_idle_timeout_s,
        overall_deadline_s=overall_deadline_s,
    )
    with _registry_lock:
        _registry[key] = policy
    return policy


def get_policy(provider: str, base_url: str, model_id: str) -> EndpointPolicy:
    """Retrieve policy for endpoint; returns permissive default if unregistered."""
    key = (_norm(provider), _norm(base_url), _norm(model_id))
    with _registry_lock:
        return _registry.get(key, _DEFAULT_POLICY)


def has_policy(provider: str, base_url: str, model_id: str) -> bool:
    """True only when an ``EndpointPolicy`` was EXPLICITLY registered for this endpoint.

    ``get_policy()`` always returns a permissive default, so callers that must not change
    behaviour for every unregistered provider have to branch on this predicate instead of
    comparing against the default values: the per-endpoint TTFT/idle policy is opt-in.
    """
    key = (_norm(provider), _norm(base_url), _norm(model_id))
    with _registry_lock:
        return key in _registry


def clear_registry() -> None:
    """Test helper — reset to empty state."""
    with _registry_lock:
        _registry.clear()


def _norm(s: str) -> str:
    return (s or "").strip().lower().rstrip("/")


# ── Production opt-in scope ───────────────────────────────────────────────────
# The endpoint policy is OPT-IN by design. In the live runtime the opt-in scope is
# the NVIDIA NIM gateway family (one gateway + one key, several model routes): every
# distinct (provider, base_url, model_id) triple routed through the NIM gateway is
# registered so the phase-aware watchdog and the endpoint-scoped breaker become
# REACHABLE in production. Non-NIM providers (Anthropic, Gemini, Bedrock, …) are
# deliberately left untouched and keep their pre-existing behaviour byte-for-byte.
NIM_HOST = "integrate.api.nvidia.com"

# Defaults applied to a NIM endpoint that has not been tuned by config.
_NIM_TTFT_TIMEOUT_S = 45.0
_NIM_CHUNK_IDLE_TIMEOUT_S = 60.0
_NIM_OVERALL_DEADLINE_S = 300.0


def is_nim_endpoint(base_url: str) -> bool:
    """True when ``base_url`` addresses the NVIDIA NIM gateway (opt-in scope)."""
    raw = (base_url or "").strip().lower()
    if not raw:
        return False
    try:
        from urllib.parse import urlparse
        netloc = urlparse(raw).netloc or raw
    except Exception:  # pragma: no cover — defensive: unknown URL shape
        netloc = raw
    netloc = netloc.split("@")[-1].split(":")[0].strip(".")
    return netloc == NIM_HOST or netloc.endswith("." + NIM_HOST)


def ensure_policy_for_endpoint(
    provider: str,
    base_url: str,
    model_id: str,
    *,
    ttft_timeout_s: float = _NIM_TTFT_TIMEOUT_S,
    chunk_idle_timeout_s: float = _NIM_CHUNK_IDLE_TIMEOUT_S,
    overall_deadline_s: float = _NIM_OVERALL_DEADLINE_S,
) -> bool:
    """Register the opt-in policy for a NIM endpoint the runtime is about to use.

    Production entry point for the opt-in gate. Idempotent: an endpoint that already
    has a registered policy is left alone (no re-registration of a stale identity,
    no overwriting of an explicitly tuned policy). Returns ``True`` only when the
    endpoint is in the NIM opt-in scope and a policy is now active for it — which is
    exactly the condition the watchdog and the breaker branch on (``has_policy``).
    """
    if not is_nim_endpoint(base_url):
        return False
    if has_policy(provider, base_url, model_id):
        return True
    register_policy(
        provider,
        base_url,
        model_id,
        ttft_timeout_s=ttft_timeout_s,
        chunk_idle_timeout_s=chunk_idle_timeout_s,
        overall_deadline_s=overall_deadline_s,
    )
    return True


__all__ = [
    "EndpointKey",
    "EndpointPolicy",
    "NIM_HOST",
    "register_policy",
    "get_policy",
    "has_policy",
    "clear_registry",
    "is_nim_endpoint",
    "ensure_policy_for_endpoint",
]