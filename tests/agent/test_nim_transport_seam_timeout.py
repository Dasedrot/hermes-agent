"""T12 — transport-level seam proof that the SDK read timeout cannot undercut the NIM idle policy.

REAL production path exercised (no parallel timeout mechanism):

1. ``agent/client_lifecycle.py:437`` ``_create_request_openai_client(reason=...)``
   -- the same function the streaming path reaches via
   ``agent/chat_completion_helpers.py:3018`` (``reason="chat_completion_stream_request"``).
2. It sets ``request_kwargs["max_retries"] = 0`` (:446) and, for an OPT-IN endpoint only,
   ``request_kwargs["timeout"] = policy.read_timeout_for_sdk()`` (:453) -- the single
   existing timeout knob (``agent/endpoint_policy.py:49``).
3. ``agent/agent_runtime_helpers.py:1913`` builds the client at the documented seam
   ``process_bootstrap.OpenAI(**client_kwargs)``.

The seam is routed over ``httpx.MockTransport`` so the assertions land on the REAL
``openai``/``httpx`` timeout object (``client._client.timeout``) -- i.e. the timeout the
transport would actually enforce -- not on a mock kwargs dict.

Scope: T12 only. No production file is modified.
"""

import contextlib

import httpx
import pytest
from openai import OpenAI as _RealOpenAI
from unittest.mock import patch

from agent import endpoint_policy as ep
from run_agent import AIAgent

NIM_BASE = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "deepseek-ai/deepseek-v4.1-flash"
NIM = ("nvidia", NIM_BASE, NIM_MODEL)
NON_NIM = ("openrouter", "https://openrouter.ai/api/v1", "test/model")

# The SDK timeout that applies when NO policy is registered (opt-in must stay a no-op).
_SDK_DEFAULT_READ = _RealOpenAI(api_key="k", base_url=NON_NIM[1])._client.timeout.read


@pytest.fixture(autouse=True)
def _isolate():
    ep.clear_registry()
    yield
    ep.clear_registry()


def _sse_transport(seen):
    """MockTransport handler: 200 + SSE for a stream, JSON otherwise. No network."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"id":"c1","object":"chat.completion.chunk","choices":'
                    '[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    return handler


class _RecordingHTTPClient(httpx.Client):
    """httpx client that records the effective per-request timeout at the send boundary.

    The openai SDK applies its configured timeout PER REQUEST, not by mutating the httpx
    client: ``openai/_base_client.py:570`` passes ``timeout=self.timeout`` into
    ``httpx.Client.build_request``, which stores it as ``request.extensions["timeout"]``
    (``httpx/_client.py:371``). ``httpx.Client.send`` only falls back to the client's own
    ``self.timeout`` when that extension is absent (``httpx/_client.py:584``) -- so the
    extension is the value the transport actually enforces for this request.
    """

    def __init__(self, *args, observed, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed = observed

    def send(self, request, **kwargs):
        ext = (getattr(request, "extensions", None) or {}).get("timeout")
        try:
            timeout = httpx.Timeout(**ext) if isinstance(ext, dict) else (ext or self.timeout)
        except Exception:  # pragma: no cover — defensive: unexpected httpx shape
            timeout = self.timeout
        if timeout is not None:
            self._observed.append(timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout))
        return super().send(request, **kwargs)


def _seam(seen, constructed, observed=None):
    """Patch target factory for ``process_bootstrap.OpenAI``.

    Passes the production kwargs through UNTOUCHED (so the real ``timeout`` /
    ``max_retries`` reach the real SDK), only replacing ``http_client`` with a
    MockTransport-backed recorder so the whole thing stays offline.
    """
    observed = observed if observed is not None else []

    def factory(**kwargs):
        injected = kwargs.pop("http_client", None)
        if injected is not None and hasattr(injected, "close"):
            with contextlib.suppress(Exception):
                injected.close()
        kwargs["http_client"] = _RecordingHTTPClient(
            transport=httpx.MockTransport(_sse_transport(seen)), observed=observed,
        )
        client = _RealOpenAI(**kwargs)
        constructed.append(client)
        return client

    return factory


def _agent(identity):
    provider, base_url, model = identity
    agent = AIAgent(
        api_key="test-key", base_url=base_url, model=model,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent.provider, agent.base_url, agent.model = provider, base_url, model
    agent._client_kwargs = {"api_key": "test-key", "base_url": base_url}
    return agent


def _sdk_read_timeout(identity, *, reason="chat_completion_stream_request"):
    """Drive the REAL production client-construction path and one real streamed request.

    Returns ``(observed, client)`` where ``observed.read`` is the read timeout the SDK handed
    the httpx transport for that request -- the value the transport would actually enforce.
    """
    from types import SimpleNamespace

    seen, constructed, observed = [], [], []
    agent = _agent(identity)
    with patch("agent.process_bootstrap.OpenAI", _seam(seen, constructed, observed)):
        client = agent._create_request_openai_client(reason=reason)
        assert client is constructed[-1], "production path did not build a real SDK client at the seam"
        list(client.chat.completions.create(
            model=identity[2], messages=[{"role": "user", "content": "hi"}], stream=True,
        ))

    assert observed, "no timeout reached the httpx transport seam"
    effective = observed[-1]
    assert effective.read is not None, "the transport was handed an unbounded read timeout"
    return SimpleNamespace(read=float(effective.read)), client


def test_T12_A_sdk_read_timeout_cannot_undercut_nim_idle_policy():
    """The real transport read timeout is >= the active NIM idle policy (and != TTFT)."""
    ep.register_policy(*NIM, ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    policy = ep.get_policy(*NIM)

    timeout, _ = _sdk_read_timeout(NIM)

    assert timeout.read == policy.read_timeout_for_sdk()
    assert timeout.read >= policy.chunk_idle_timeout_s, (
        f"SDK read timeout {timeout.read} undercuts idle policy {policy.chunk_idle_timeout_s}"
    )
    assert timeout.read > policy.ttft_timeout_s, "SDK timeout must not equal the TTFT budget"


@pytest.mark.parametrize(
    "idle,expected",
    [(15.0, 45.0), (60.0, 90.0), (120.0, 150.0), (700.0, 730.0)],
)
def test_T12_B_idle_policy_change_moves_the_transport_timeout(idle, expected):
    """Changing the policy idle value changes the REAL httpx timeout accordingly."""
    ep.register_policy(*NIM, ttft_timeout_s=5.0, chunk_idle_timeout_s=idle, overall_deadline_s=1000.0)

    timeout, _ = _sdk_read_timeout(NIM)

    assert timeout.read == expected
    assert timeout.read >= idle


def test_T12_C_invariant_guard_fails_when_sdk_timeout_would_undercut_idle():
    """Falsification guard for req. 6: any read timeout below the idle policy is a failure.

    Sweeps idle both below and above the TTFT budget so a regression of the form
    ``timeout = ttft`` (or a bare SDK default) is caught for every shape.
    """
    for idle in (5.0, 45.0, 60.0, 120.0, 700.0):
        ep.clear_registry()
        ep.register_policy(*NIM, ttft_timeout_s=45.0, chunk_idle_timeout_s=idle, overall_deadline_s=1000.0)
        policy = ep.get_policy(*NIM)

        timeout, _ = _sdk_read_timeout(NIM)

        assert timeout.read >= policy.chunk_idle_timeout_s, (
            f"idle={idle}: transport read timeout {timeout.read} undercuts the idle policy"
        )
        if idle > _SDK_DEFAULT_READ:
            assert timeout.read > _SDK_DEFAULT_READ, (
                f"idle={idle} exceeds the SDK default {_SDK_DEFAULT_READ}; the policy must raise it"
            )


def test_T12_D_non_nim_endpoint_keeps_sdk_default():
    """Opt-in: an endpoint without a registered policy keeps the untouched SDK default."""
    assert ep.has_policy(*NON_NIM) is False

    timeout, _ = _sdk_read_timeout(NON_NIM)

    assert timeout.read == _SDK_DEFAULT_READ, (
        f"non-opt-in endpoint timeout changed: {timeout.read} != SDK default {_SDK_DEFAULT_READ}"
    )


def test_T12_E_streaming_request_completes_over_the_real_transport_seam():
    """The client built by production is functional: a real streamed call round-trips.

    Also asserts the effective per-request timeout at the send boundary matches the policy
    (the same observable as the tests above, driven here to completion).
    """
    ep.register_policy(*NIM, ttft_timeout_s=45.0, chunk_idle_timeout_s=60.0, overall_deadline_s=300.0)
    seen, constructed, observed = [], [], []
    agent = _agent(NIM)

    with patch("agent.process_bootstrap.OpenAI", _seam(seen, constructed, observed)):
        client = agent._create_request_openai_client(reason="chat_completion_stream_request")
        chunks = list(client.chat.completions.create(
            model=NIM_MODEL, messages=[{"role": "user", "content": "hi"}], stream=True,
        ))

    assert seen, "no request reached the transport seam"
    assert chunks and chunks[-1].choices[0].delta.content == "ok"
    assert observed, "no timeout reached the httpx transport seam"
    effective_read = float(observed[-1].read)
    policy = ep.get_policy(*NIM)
    assert effective_read == policy.read_timeout_for_sdk()
    assert effective_read >= policy.chunk_idle_timeout_s
