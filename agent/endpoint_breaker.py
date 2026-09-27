"""Endpoint-scoped circuit breaker (pure state machine, no IO).

States:
    CLOSED    -> requests allowed
    OPEN      -> requests rejected immediately until cooldown expires
    HALF_OPEN -> exactly ONE probe allowed; others skip

Breaker key: (provider, base_url, model_id) — same as EndpointPolicy.

Thread-safe, process-local. Public reset for test isolation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Set
from contextlib import contextmanager


EndpointKey = Tuple[str, str, str]  # (provider, base_url, model_id)


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    """Configuration for a circuit breaker."""
    cooldown_s: float = 300.0  # 5 minutes default

    def __post_init__(self) -> None:
        if self.cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")


@dataclass(slots=True)
class BreakerSnapshot:
    """Immutable view of breaker state for logging/inspection."""
    state: BreakerState
    last_failure_at: Optional[float]
    half_open_probe_in_flight: bool
    open_until: Optional[float]

    def __str__(self) -> str:
        if self.state == BreakerState.OPEN:
            rem = max(0.0, (self.open_until or 0) - time.monotonic())
            return f"OPEN (cooldown {rem:.0f}s)"
        if self.state == BreakerState.HALF_OPEN:
            return f"HALF_OPEN (probe={'in_flight' if self.half_open_probe_in_flight else 'idle'})"
        return "CLOSED"


@dataclass(slots=True)
class _BreakerEntry:
    """Mutable internal state for a single endpoint breaker."""
    state: BreakerState = BreakerState.CLOSED
    last_failure_at: Optional[float] = None
    half_open_probe_in_flight: bool = False
    open_until: Optional[float] = None
    config: BreakerConfig = field(default_factory=BreakerConfig)

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            state=self.state,
            last_failure_at=self.last_failure_at,
            half_open_probe_in_flight=self.half_open_probe_in_flight,
            open_until=self.open_until,
        )


# Global registry — thread-safe, process-local
_registry: Dict[EndpointKey, _BreakerEntry] = {}
_registry_lock = threading.RLock()

# Default config
_DEFAULT_CONFIG = BreakerConfig()


def _norm(s: str) -> str:
    return (s or "").strip().lower().rstrip("/")


def _key(provider: str, base_url: str, model_id: str) -> EndpointKey:
    return (_norm(provider), _norm(base_url), _norm(model_id))


def get_breaker(provider: str, base_url: str, model_id: str) -> _BreakerEntry:
    """Get or create breaker entry for endpoint."""
    k = _key(provider, base_url, model_id)
    with _registry_lock:
        entry = _registry.get(k)
        if entry is None:
            entry = _BreakerEntry()
            _registry[k] = entry
        return entry


def record_success(provider: str, base_url: str, model_id: str) -> BreakerSnapshot:
    """Record successful completion — only from CLOSED or HALF_OPEN."""
    entry = get_breaker(provider, base_url, model_id)
    with _registry_lock:
        if entry.state == BreakerState.HALF_OPEN:
            # Successful probe -> CLOSED
            entry.state = BreakerState.CLOSED
            entry.half_open_probe_in_flight = False
            entry.open_until = None
            entry.last_failure_at = None
        # CLOSED stays CLOSED (success doesn't change state)
        return entry.snapshot()


def record_failure(
    provider: str,
    base_url: str,
    model_id: str,
    *,
    qualifies: bool,
) -> BreakerSnapshot:
    """Record a failure that may trip the breaker.

    Args:
        qualifies: True if this failure class should trip the breaker
                   (transport/5xx/timeouts). False for 400/401/403/404/422.
    """
    entry = get_breaker(provider, base_url, model_id)
    now = time.monotonic()
    with _registry_lock:
        if entry.state == BreakerState.HALF_OPEN:
            # Probe failed — what happens next depends on failure type
            entry.half_open_probe_in_flight = False
            if qualifies:
                # Qualifying failure -> back to OPEN
                entry.state = BreakerState.OPEN
                entry.open_until = now + entry.config.cooldown_s
                entry.last_failure_at = now
            else:
                # Non-qualifying (config/client error) -> CLOSED
                # Don't punish the endpoint for our misconfiguration
                entry.state = BreakerState.CLOSED
                entry.last_failure_at = None
                entry.open_until = None
            return entry.snapshot()

        if entry.state == BreakerState.CLOSED:
            if qualifies:
                # First qualifying failure -> OPEN
                entry.state = BreakerState.OPEN
                entry.open_until = now + entry.config.cooldown_s
                entry.last_failure_at = now
            # Non-qualifying in CLOSED -> stay CLOSED
            return entry.snapshot()

        # OPEN -> just refresh the failure timestamp
        if qualifies:
            entry.last_failure_at = now
            entry.open_until = now + entry.config.cooldown_s
        return entry.snapshot()


def can_attempt(provider: str, base_url: str, model_id: str) -> bool:
    """Check if a request may proceed (consumes HALF_OPEN probe slot)."""
    entry = get_breaker(provider, base_url, model_id)
    now = time.monotonic()
    with _registry_lock:
        if entry.state == BreakerState.CLOSED:
            return True

        if entry.state == BreakerState.OPEN:
            if now >= (entry.open_until or 0):
                # Cooldown expired -> transition to HALF_OPEN for this probe
                entry.state = BreakerState.HALF_OPEN
                entry.half_open_probe_in_flight = True
                entry.open_until = None
                return True
            return False  # still in cooldown

        # HALF_OPEN
        if entry.half_open_probe_in_flight:
            return False  # another probe already in flight
        # First caller gets the probe slot
        entry.half_open_probe_in_flight = True
        return True


def release_probe(provider: str, base_url: str, model_id: str) -> None:
    """Release a reserved HALF_OPEN probe WITHOUT recording success or failure.

    Caller cancellation is not an endpoint-health signal: the request never got to prove
    anything about the endpoint, so the breaker must neither trip nor be credited. It must
    only stop holding the single probe slot, or the endpoint is stuck in HALF_OPEN forever
    and every later caller is refused (violates the "no path leaves a probe reserved"
    invariant). Terminal state is left as-is (HALF_OPEN), so the next caller may probe.
    """
    entry = get_breaker(provider, base_url, model_id)
    with _registry_lock:
        if entry.state == BreakerState.HALF_OPEN:
            entry.half_open_probe_in_flight = False


@contextmanager
def attempt_slot(provider: str, base_url: str, model_id: str):
    """Context manager that reserves a HALF_OPEN slot and releases on exit.

    Usage:
        if not can_attempt(...):
            raise SkipEndpoint("breaker open")
        with attempt_slot(...):
            try:
                yield
                record_success(...)
            except Exception as e:
                record_failure(..., qualifies=should_trip(e))
                raise

    If the attempt fails with a qualifying error, the breaker will be updated
    by record_failure. If it succeeds, record_success transitions HALF_OPEN->CLOSED.
    """
    k = _key(provider, base_url, model_id)
    entry = get_breaker(provider, base_url, model_id)
    now = time.monotonic()
    acquired = False
    with _registry_lock:
        if entry.state == BreakerState.HALF_OPEN and not entry.half_open_probe_in_flight:
            entry.half_open_probe_in_flight = True
            acquired = True
        elif entry.state == BreakerState.OPEN and now >= (entry.open_until or 0):
            entry.state = BreakerState.HALF_OPEN
            entry.half_open_probe_in_flight = True
            entry.open_until = None
            acquired = True
        elif entry.state == BreakerState.CLOSED:
            acquired = True  # CLOSED doesn't use probe slot

    if not acquired:
        raise RuntimeError(f"Breaker {k} not available (OPEN/HALF_OPEN with probe in flight)")

    try:
        yield
    finally:
        if acquired:
            with _registry_lock:
                if entry.state == BreakerState.HALF_OPEN:
                    entry.half_open_probe_in_flight = False


def get_state(provider: str, base_url: str, model_id: str) -> BreakerSnapshot:
    """Get current breaker state (read-only)."""
    return get_breaker(provider, base_url, model_id).snapshot()


def set_config(provider: str, base_url: str, model_id: str, config: BreakerConfig) -> None:
    """Override breaker config for a specific endpoint."""
    entry = get_breaker(provider, base_url, model_id)
    with _registry_lock:
        entry.config = config


def clear_registry() -> None:
    """Test helper — reset all breakers."""
    with _registry_lock:
        _registry.clear()


__all__ = [
    "BreakerState",
    "BreakerConfig",
    "BreakerSnapshot",
    "record_success",
    "record_failure",
    "can_attempt",
    "release_probe",
    "attempt_slot",
    "get_state",
    "set_config",
    "clear_registry",
]