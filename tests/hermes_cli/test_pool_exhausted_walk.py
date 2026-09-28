"""D9 incident-scenario test: NVIDIA pool exhausted → OpenRouter lane #5 → paid DeepSeek.

This test verifies the exact fallback cascade motivated by the OpenRouter lane addition.
No live model/API calls — mocked transport seam only.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import _candidate_pool_exhausted, try_activate_fallback
from agent.credential_pool import load_pool, PooledCredential, STATUS_EXHAUSTED
from run_agent import AIAgent


ARCHITECT_CHAIN = [
    {"provider": "nvidia", "model": "z-ai/glm-5.3"},
    {"provider": "nvidia", "model": "deepseek-ai/deepseek-v4.1-flash"},
    {"provider": "nvidia", "model": "moonshotai/kimi-k3"},
    {"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "base_url": "https://openrouter.ai/api/v1"},
    {"provider": "deepseek", "model": "deepseek-flash"},
]


def _make_agent_with_architect_chain():
    """Create an AIAgent with architect's exact fallback chain."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            provider="nvidia",
            model="nvidia/nemotron-3-ultra-550b-a55b",
            api_key="test-primary-key",
            base_url="https://integrate.api.nvidia.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=ARCHITECT_CHAIN,
        )
        agent.client = MagicMock()
        agent.client.base_url = "https://integrate.api.nvidia.com/v1"
        agent.client.api_key = "test-primary-key"
        return agent


def _exhaust_nvidia_pool(agent):
    """Mark the NVIDIA credential pool as exhausted (all credentials in long cooldown)."""
    pool = load_pool("nvidia")
    # Ensure pool has entries (tests run with potentially empty pool)
    if not pool._entries:
        pool._entries = [
            PooledCredential(
                provider="nvidia",
                id="nvidia-test-1",
                label="nvidia-test-1",
                auth_type="api_key",
                source="test",
                base_url="https://integrate.api.nvidia.com/v1",
                access_token="test-key-1",
                priority=0,
                last_status=STATUS_EXHAUSTED,
                last_status_at=time.time() - 100,
                last_error_reset_at=time.time() + 1200,  # >600s = exhausted
            ),
            PooledCredential(
                provider="nvidia",
                id="nvidia-test-2",
                label="nvidia-test-2",
                auth_type="api_key",
                source="test",
                base_url="https://integrate.api.nvidia.com/v1",
                access_token="test-key-2",
                priority=0,
                last_status=STATUS_EXHAUSTED,
                last_status_at=time.time() - 100,
                last_error_reset_at=time.time() + 1200,
            ),
        ]
    else:
        for entry in pool._entries:
            entry.last_status = STATUS_EXHAUSTED
            entry.last_status_at = time.time() - 100
            entry.last_error_reset_at = time.time() + 1200
    # Persist so subsequent load_pool() calls see the exhausted state
    from agent.credential_pool import persist_pool_entries
    persist_pool_entries(pool.provider, [e.to_dict() for e in pool._entries])
    # Also attach to agent so _candidate_pool_exhausted uses this instance
    agent._credential_pool = pool
    return pool


def _verify_openrouter_pool_consulted():
    """Verify that OpenRouter resolution uses only the openrouter pool."""
    from agent import auxiliary_client

    asked = []

    def _record(provider: str):
        asked.append(provider)
        return False, None

    client = object()
    with (
        patch.object(auxiliary_client, "_select_pool_entry", _record),
        patch.object(auxiliary_client, "_scoped_key_env", lambda name: "test-key"),
        patch.object(auxiliary_client, "_create_openai_client", lambda **kw: client),
    ):
        resolved, model = auxiliary_client._try_openrouter(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            explicit_base_url="https://openrouter.ai/api/v1",
        )

    assert asked == ["openrouter"], f"Expected only openrouter pool, got {asked}"
    assert resolved is client
    assert model == "nvidia/nemotron-3-ultra-550b-a55b:free"


class TestD9IncidentScenario:
    """Full fallback cascade: exhausted NVIDIA pool → OpenRouter → paid DeepSeek."""

    def test_nvidia_pool_exhausted_reaches_openrouter_then_deepseek(self):
        """
        Scenario:
        1. Primary architect model: direct NVIDIA Nemotron
        2. NVIDIA credential pool is exhausted before fallback walk
        3. Execute fallback cascade via mocked transport seam
        4. NVIDIA lanes 2-4 skipped by credential gate
        5. Cascade ATTEMPTS OpenRouter lane #5 using OpenRouter pool only
        6. OpenRouter fails → cascade proceeds to paid DeepSeek lane #6
        7. Lane #3 (deepseek-ai/deepseek-v4.1-flash) remains in configured order
        """
        agent = _make_agent_with_architect_chain()

        # Verify initial state
        assert agent.provider == "nvidia"
        assert agent.model == "nvidia/nemotron-3-ultra-550b-a55b"
        assert len(agent._fallback_chain) == 5

        # Exhaust the NVIDIA pool
        _exhaust_nvidia_pool(agent)

        # Verify pool exhaustion logic works for each NVIDIA lane
        assert _candidate_pool_exhausted(agent, "nvidia", "z-ai/glm-5.3") is True
        assert _candidate_pool_exhausted(agent, "nvidia", "deepseek-ai/deepseek-v4.1-flash") is True
        assert _candidate_pool_exhausted(agent, "nvidia", "moonshotai/kimi-k3") is True

        # Verify OpenRouter pool is NOT exhausted (fresh)
        assert _candidate_pool_exhausted(agent, "openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free") is False

        # Verify DeepSeek pool is NOT exhausted
        assert _candidate_pool_exhausted(agent, "deepseek", "deepseek-flash") is False

        # Verify lane #3 (deepseek-ai/deepseek-v4.1-flash) remains in configured order
        lane3 = agent._fallback_chain[1]
        assert lane3["provider"] == "nvidia"
        assert lane3["model"] == "deepseek-ai/deepseek-v4.1-flash"

        # Simulate fallback walk with mocked transport
        # Track which lanes were attempted
        attempted = []

        def mock_resolve(provider, model=None, raw_codex=False, explicit_base_url=None, explicit_api_key=None, api_mode=None):
            attempted.append((provider, model))
            if provider == "nvidia":
                return None, None  # exhausted pool → no client
            if provider == "openrouter":
                return None, None  # simulate OpenRouter failure
            if provider == "deepseek":
                mock_client = MagicMock()
                mock_client.base_url = "https://api.deepseek.com/v1"
                mock_client.api_key = "deepseek-key"
                return mock_client, "deepseek-flash"
            return None, None

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=mock_resolve):
            from agent.error_classifier import FailoverReason

            # Walk the fallback chain by calling try_activate_fallback repeatedly
            # until chain exhausted or success. try_activate_fallback returns True
            # when a fallback is successfully activated, False when chain exhausted.
            # Stop on first successful activation.
            max_calls = len(agent._fallback_chain) + 2
            for _ in range(max_calls):
                result = try_activate_fallback(agent, FailoverReason.rate_limit)
                if not result:
                    break
                # result=True means a fallback was successfully activated - stop here
                break

        # Verify the walk order
        nvidia_attempts = [a for a in attempted if a[0] == "nvidia"]
        openrouter_attempts = [a for a in attempted if a[0] == "openrouter"]
        deepseek_attempts = [a for a in attempted if a[0] == "deepseek"]

        # NVIDIA lanes 2-4 should be SKIPPED by credential gate (pool exhausted)
        # The gate _should_skip_fallback_candidate checks _candidate_pool_exhausted first
        # so they should not even attempt resolution
        assert len(nvidia_attempts) == 0, f"NVIDIA lanes should be skipped by credential gate, got {nvidia_attempts}"

        # OpenRouter lane #5 MUST be attempted
        assert len(openrouter_attempts) == 1, f"OpenRouter must be attempted once, got {openrouter_attempts}"
        assert openrouter_attempts[0][1] == "nvidia/nemotron-3-ultra-550b-a55b:free"

        # DeepSeek lane #6 MUST be attempted after OpenRouter failure
        assert len(deepseek_attempts) == 1, f"DeepSeek must be attempted once, got {deepseek_attempts}"

        # Verify OpenRouter uses its own pool
        _verify_openrouter_pool_consulted()

    def test_openrouter_success_stops_before_deepseek(self):
        """If OpenRouter lane succeeds, DeepSeek lane is never attempted."""
        agent = _make_agent_with_architect_chain()
        _exhaust_nvidia_pool(agent)

        attempted = []

        def mock_resolve(provider, model=None, raw_codex=False, explicit_base_url=None, explicit_api_key=None, api_mode=None):
            attempted.append((provider, model))
            if provider == "nvidia":
                return None, None
            if provider == "openrouter":
                mock_client = MagicMock()
                mock_client.base_url = "https://openrouter.ai/api/v1"
                mock_client.api_key = "or-key"
                return mock_client, "nvidia/nemotron-3-ultra-550b-a55b:free"
            if provider == "deepseek":
                # Should never reach here
                raise AssertionError("DeepSeek should not be attempted when OpenRouter succeeds")
            return None, None

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=mock_resolve):
            from agent.error_classifier import FailoverReason

            # Walk the fallback chain - stop on first successful activation
            max_calls = len(agent._fallback_chain) + 2
            for _ in range(max_calls):
                result = try_activate_fallback(agent, FailoverReason.rate_limit)
                if not result:
                    break
                # result=True means a fallback was successfully activated - stop here
                break

        # Verify only OpenRouter was attempted after NVIDIA
        openrouter_attempts = [a for a in attempted if a[0] == "openrouter"]
        deepseek_attempts = [a for a in attempted if a[0] == "deepseek"]

        assert len(openrouter_attempts) == 1
        assert len(deepseek_attempts) == 0, "DeepSeek must not be attempted when OpenRouter succeeds"

        # Verify OpenRouter pool separation
        _verify_openrouter_pool_consulted()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])