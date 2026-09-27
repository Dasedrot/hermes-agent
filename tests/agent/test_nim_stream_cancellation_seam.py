"""§3 A/B/C — cancellation, early abandonment and mid-stream failure through the REAL
``_StreamingCall`` seam.

Driven like the existing streaming suite (tests/agent/test_streaming.py):
``agent._interruptible_streaming_api_call({}, on_first_delta=...)`` ->
``_StreamingCall(...).run()``, with ``AIAgent._create_request_openai_client`` patched to a
client whose ``chat.completions.create()`` returns a controllable provider stream.
No second streaming implementation; no production change.
"""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import endpoint_breaker as eb
from agent import endpoint_policy as ep
from run_agent import AIAgent

NIM = ("nvidia", "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4.1-flash")


def _chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="nim-model", usage=None)


@pytest.fixture(autouse=True)
def _isolate():
    eb.clear_registry()
    ep.clear_registry()
    yield
    eb.clear_registry()
    ep.clear_registry()


def _agent():
    agent = AIAgent(
        api_key="test-key", base_url=NIM[1], model=NIM[2],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent.provider, agent.base_url, agent.model = NIM
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _arm_half_open_probe():
    """Trip the breaker, let the cooldown lapse, then reserve the HALF_OPEN probe."""
    eb.set_config(*NIM, eb.BreakerConfig(cooldown_s=0.01))
    eb.record_failure(*NIM, qualifies=True)
    time.sleep(0.02)
    admitted = eb.can_attempt(*NIM)
    snap = eb.get_state(*NIM)
    assert admitted and snap.half_open_probe_in_flight, f"probe not armed: {snap}"
    return snap


def _run_call(agent, client, *, on_first_delta=None):
    """Drive the real streaming seam with a patched per-request client."""
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=client), \
         patch("run_agent.AIAgent._close_request_openai_client"):
        return agent._interruptible_streaming_api_call({}, on_first_delta=on_first_delta)


def _wait_probe_released(timeout=8.0) -> bool:
    """The worker unwinds on its own thread; wait (bounded) for the probe to be released."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not eb.get_state(*NIM).half_open_probe_in_flight:
            return True
        time.sleep(0.05)
    return False


def test_A_caller_cancellation_is_prompt_and_releases_the_probe_without_opening():
    """A: cancel while the stream is waiting -> prompt, probe released, breaker NOT opened."""
    ep.register_policy(*NIM, ttft_timeout_s=60.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    _arm_half_open_probe()
    agent = _agent()
    started = threading.Event()

    def blocking_stream():
        started.set()
        while not agent._interrupt_requested:
            time.sleep(0.02)
        # Let the monitor's 0.3s loop observe the flag and mark the attempt cancelled;
        # that is what makes the worker classify this as a cancel, not a network error.
        time.sleep(0.6)
        raise httpx.ReadError("force-closed by interrupt (caller cancel)")
        yield  # unreachable: makes this a lazy generator

    client = MagicMock()
    client.chat.completions.create.return_value = blocking_stream()

    def cancel():
        assert started.wait(5.0), "stream never started"
        time.sleep(0.2)
        agent._interrupt_requested = True

    threading.Thread(target=cancel, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(InterruptedError):
        _run_call(agent, client)
    elapsed = time.monotonic() - t0

    assert elapsed < 10.0, f"cancellation was not prompt: {elapsed:.2f}s"
    assert _wait_probe_released(), "HALF_OPEN probe was never released after cancellation"
    snap = eb.get_state(*NIM)
    assert snap.state is not eb.BreakerState.OPEN, "caller cancellation opened the breaker"
    assert snap.state is eb.BreakerState.HALF_OPEN, f"unexpected state after cancel: {snap}"
    assert snap.half_open_probe_in_flight is False
    assert eb.can_attempt(*NIM) is True, "endpoint still refuses callers after a cancel"


class _OneEventStream:
    """Provider stream that yields ONE event, then ends because the consumer abandoned it."""

    def __init__(self, on_close):
        self._on_close = on_close
        self._yielded = False
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._yielded:
            self._yielded = True
            return _chunk(content="partial")
        raise StopIteration

    def close(self):
        self.closed = True
        self._on_close()


def test_B_early_consumer_abandonment_is_clean_and_not_a_failure():
    """B: one meaningful event, then abandon via the real close path -> no failure recorded."""
    ep.register_policy(*NIM, ttft_timeout_s=60.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    _arm_half_open_probe()
    agent = _agent()
    seen = []
    stream = _OneEventStream(lambda: None)
    client = MagicMock()
    client.chat.completions.create.return_value = stream

    response = _run_call(agent, client, on_first_delta=lambda *a: seen.append(a))

    # Shut down cleanly through the real managed-stream close path.
    assert stream.closed is True, "provider stream was never closed by the real close path"
    assert len(seen) >= 1, "no consumer event was delivered before abandonment"
    assert _wait_probe_released(), "probe left reserved after a clean abandonment"
    snap = eb.get_state(*NIM)
    assert snap.state is not eb.BreakerState.OPEN, "abandonment falsely opened the breaker"
    assert snap.state is eb.BreakerState.CLOSED, f"unexpected state: {snap}"
    assert snap.half_open_probe_in_flight is False
    assert response is not None, "seam produced no response object at all"
    assert response.choices[0].message.content == "partial"


def test_C_mid_stream_provider_failure_after_output_does_not_fall_back():
    """C: event delivered, then the real stream fails -> no fallback, failure recorded, no dupes."""
    ep.register_policy(*NIM, ttft_timeout_s=60.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    _arm_half_open_probe()
    agent = _agent()
    seen = []
    create_calls = []
    fallback_calls = []

    def failing_stream():
        yield _chunk(content="partial text")
        # A real display consumer recorded this text before the connection died.
        agent._record_streamed_assistant_text("partial text")
        raise httpx.ReadError("connection reset mid-stream")

    def create(**kwargs):
        create_calls.append(kwargs)
        return failing_stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = create

    def counting_fallback(self, *args, **kwargs):
        fallback_calls.append((args, kwargs))
        return False

    with patch.object(AIAgent, "_try_activate_fallback", counting_fallback):
        response = _run_call(agent, client, on_first_delta=lambda *a: seen.append(a))

    assert fallback_calls == [], f"fell back to another model after output began: {fallback_calls}"
    assert len(create_calls) == 1, f"stream was retried after output began ({len(create_calls)} attempts)"
    snap = eb.get_state(*NIM)
    assert snap.state is eb.BreakerState.OPEN, f"provider failure was not recorded: {snap}"
    assert snap.half_open_probe_in_flight is False
    assert response is not None, "expected a partial stub, got no response"
    text = response.choices[0].message.content or ""
    assert text.count("partial text") == 1, f"partial output duplicated/concatenated: {text!r}"





