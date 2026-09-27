"""Display and heartbeat phase of the request-local streaming monitor."""

import time
from types import SimpleNamespace

from agent import chat_completion_wait_notice as wn
from agent.model_metadata import is_local_endpoint


class StreamingWaitMonitor:
    def _poll_local_load_notice(self, now: float) -> bool:
        """Managed local server: surface a cold model's weight-load progress
        instead of the 60s neutral "waiting on <model>" notice. Polled ~1s only while no
        REAL chunk arrived for 2s+ (never during healthy token flow); in-memory,
        no network. True while loading = heartbeat liveness, skip the rest of
        this iteration (the stale detector's local floor dwarfs any load)."""
        from agent.chat_completion_helpers import _managed_local_load_notice

        m = self._mon
        if now - self.last_chunk_time["t"] < 2.0 or now - m.last_load_poll < 1.0:
            return False
        m.last_load_poll = now
        _load_notice = _managed_local_load_notice(self.agent, self.api_kwargs)
        if _load_notice is not None:
            m.wait_notice_started_ts = None  # The local loader now owns the display.
            m.wait_notice.reset()
            self.agent._emit_wait_notice(_load_notice)
            self.agent._touch_activity("local model loading")
            m.load_notice_shown, m.load_notice_misses, m.last_heartbeat = True, 0, now  # loading IS liveness
            return True
        if m.load_notice_shown:
            # One missed sample is routine (probe timeout under load); clearing on it strobed the line.
            m.load_notice_misses += 1
            if m.load_notice_misses >= 3:
                m.load_notice_shown, m.load_notice_misses = False, 0
                self.agent._emit_wait_notice("")
        return False

    def _effective_stale_timeout(self) -> float:
        """Phase-aware stale threshold for the CURRENT attempt (the single existing
        watchdog threshold — no second detector).

        Endpoints opt in by having an EXPLICITLY registered ``EndpointPolicy``
        (``self._endpoint_policy``, resolved by ``_resolve_stale_timeout``); for every
        other endpoint this returns ``self._stream_stale_timeout`` unchanged, so the
        stale-kill threshold they get is byte-for-byte the pre-existing one.

        The phase is the same signal this monitor already uses to label the wait
        (``diag["first_chunk_at"]``, stamped by ``_count_chunk`` on the first real
        chunk): before it the attempt may wait ``policy.ttft_timeout_s`` for the first
        event, after it the tolerated silence is ``policy.chunk_idle_timeout_s``. Both
        are bounded by the cascade's shared deadline and by the pre-existing per-provider
        stale timeout (see ``_policy_phase_stale_timeout``).
        """
        base = getattr(self, "_stream_stale_timeout", None)
        policy = getattr(self, "_endpoint_policy", None)
        if policy is None:
            return float("inf") if base is None else base
        try:
            from agent.chat_completion_helpers import _policy_phase_stale_timeout
            diag = getattr(getattr(self, "clients", None), "diag", None)
            first_chunk_at = diag.get("first_chunk_at") if isinstance(diag, dict) else None
            return _policy_phase_stale_timeout(
                policy, base, first_chunk_seen=bool(first_chunk_at),
                deadline=getattr(self, "_cascade_deadline", None), now=time.monotonic())
        except Exception:
            # Never let the policy layer change the safety property of the watchdog.
            import logging
            logging.getLogger(__name__).debug("Phase-aware stale timeout failed; using the base threshold", exc_info=True)
            return float("inf") if base is None else base

    def _heartbeat(self, waiting_secs: int) -> None:
        """Gateway inactivity heartbeat: the start-to-first-chunk gap (thinking,
        local prefill) can exceed the gateway timeout."""
        if waiting_secs >= 60.0:
            # No chunks for 60s+: say WHAT the wait is and WHEN recovery kicks in —
            # once per silence, not every heartbeat (#92550).
            stale = self._effective_stale_timeout()
            watchdog = ("stream stale", stale - waiting_secs) if stale is not None and stale != float("inf") else None
            diag = getattr(getattr(self, "clients", None), "diag", None)
            phase = "post_chunk" if isinstance(diag, dict) and diag.get("first_chunk_at") else "first_chunk"
            if not self._mon.wait_notice.should_emit(phase, watchdog):
                self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, {phase})")
                return
            self._mon.wait_notice_started_ts = self._mon.last_heartbeat
            self.agent._emit_wait_notice(wn.wait_notice_text(
                self.api_kwargs.get('model', 'the provider'), waiting_secs, phase, watchdog))
        else:
            # Chunks are flowing — keep the tracker fresh, leave the display alone.
            self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, no chunks yet)")

    def _monitor_loop(self) -> None:
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        self._mon = SimpleNamespace(
            last_heartbeat=time.time(), last_load_poll=0.0,
            load_notice_shown=False, load_notice_misses=0, wait_notice_started_ts=None,
            wait_notice=wn.WaitNoticeState(),
        )
        _is_local_base = bool(self.agent.base_url) and is_local_endpoint(self.agent.base_url)
        while not self._call_done.is_set():
            self._call_done.wait(timeout=0.3)
            _hb_now = time.time()
            if _is_local_base and self._poll_local_load_notice(_hb_now):
                continue
            # Reasoning callbacks do not clear the classic CLI spinner. The empty
            # protocol payload resets status without adding synthetic reasoning.
            if (self._mon.wait_notice_started_ts is not None
                    and self.last_chunk_time["t"] > self._mon.wait_notice_started_ts):
                self.agent._emit_wait_notice("")
                self._mon.wait_notice_started_ts = None
                self._mon.wait_notice.reset()
            if _hb_now - self._mon.last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._mon.last_heartbeat = _hb_now
                self._heartbeat(int(_hb_now - self.last_chunk_time["t"]))
            _stale_elapsed = time.time() - self.last_chunk_time["t"]
            if _stale_elapsed > self._effective_stale_timeout():
                self._mon.wait_notice_started_ts = None  # Reconnect status has its own owner.
                self._mon.wait_notice.reset()
                self._kill_stale_stream(_stale_elapsed)
            if self.agent._interrupt_requested:
                self._abort_for_interrupt(_stale_elapsed)
                return

