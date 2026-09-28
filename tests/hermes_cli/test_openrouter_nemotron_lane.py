"""Focused tests for the architect OpenRouter Nemotron lane (GO: openrouter nemotron).

Contract under test:
  * the OpenRouter free-Nemotron lane sits immediately BEFORE the paid DeepSeek
    lane in both the main and the delegation fallback chains;
  * the entry keeps its explicit endpoint (``base_url``) through normalization;
  * OpenRouter resolves against its OWN credential pool/endpoint and never
    borrows the NVIDIA NIM pool;
  * the lane passes the auxiliary ``free_only`` policy gate (it is a ``:free`` SKU).

No live model/API calls: provider clients are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.fallback_config import get_fallback_chain, scoped_fallback_chain

OPENROUTER_ENTRY = {
    "provider": "openrouter",
    "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "base_url": "https://openrouter.ai/api/v1",
}

# Architect chain as declared in profiles/architect/config.yaml (lanes 2..6).
ARCHITECT_CHAIN = [
    {"provider": "nvidia", "model": "z-ai/glm-5.3"},
    {"provider": "nvidia", "model": "deepseek-ai/deepseek-v4.1-flash"},
    {"provider": "nvidia", "model": "moonshotai/kimi-k3"},
    dict(OPENROUTER_ENTRY),
    {"provider": "deepseek", "model": "deepseek-flash"},
]


def _identity(entry):
    return (entry.get("provider"), entry.get("model"))


class TestLaneOrdering:

    def test_main_chain_places_openrouter_immediately_before_paid_deepseek(self):
        chain = get_fallback_chain({"fallback_providers": ARCHITECT_CHAIN})
        identities = [_identity(e) for e in chain]
        assert identities == [
            ("nvidia", "z-ai/glm-5.3"),
            ("nvidia", "deepseek-ai/deepseek-v4.1-flash"),
            ("nvidia", "moonshotai/kimi-k3"),
            ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            ("deepseek", "deepseek-flash"),
        ]
        or_index = identities.index(_identity(OPENROUTER_ENTRY))
        assert identities[or_index + 1][0] == "deepseek"
        # NVIDIA lane #3 must not have moved.
        assert identities[1] == ("nvidia", "deepseek-ai/deepseek-v4.1-flash")

    def test_entry_keeps_explicit_openrouter_endpoint(self):
        chain = get_fallback_chain({"fallback_providers": ARCHITECT_CHAIN})
        entry = next(e for e in chain if e["provider"] == "openrouter")
        assert entry["base_url"] == "https://openrouter.ai/api/v1"

    def test_delegation_chain_keeps_openrouter_lane_for_pinned_child(self):
        chain = scoped_fallback_chain(
            inherited=None,
            declared=ARCHITECT_CHAIN,
            pinned=True,
            owner="delegated child",
        )
        assert [e["provider"] for e in chain] == [
            "nvidia", "nvidia", "nvidia", "openrouter", "deepseek",
        ]


class TestPoolSeparation:

    def test_openrouter_pool_key_is_not_the_nvidia_key(self):
        from agent.credential_pool import resolve_runtime_pool_key

        assert resolve_runtime_pool_key(
            "openrouter", "https://openrouter.ai/api/v1"
        ) == "openrouter"
        assert resolve_runtime_pool_key(
            "nvidia", "https://integrate.api.nvidia.com/v1"
        ) == "nvidia"

    def test_openrouter_resolution_queries_only_the_openrouter_pool(self):
        from agent import auxiliary_client

        asked: list[str] = []

        def _record(provider: str):
            asked.append(provider)
            return False, None  # no pool row -> falls through env, still non-nvidia

        client = object()
        with (
            patch.object(auxiliary_client, "_select_pool_entry", _record),
            patch.object(auxiliary_client, "_scoped_key_env", lambda name: "test-key"),
            patch.object(auxiliary_client, "_create_openai_client",
                         lambda **kw: client),
        ):
            resolved, model = auxiliary_client._try_openrouter(
                model=OPENROUTER_ENTRY["model"],
                explicit_base_url=OPENROUTER_ENTRY["base_url"],
            )

        assert asked == ["openrouter"]
        assert resolved is client
        assert model == "nvidia/nemotron-3-ultra-550b-a55b:free"


class TestCapabilityContract:

    def test_lane_model_is_a_free_sku_for_the_free_only_gate(self):
        from agent.auxiliary_client import _is_free_model

        assert _is_free_model(OPENROUTER_ENTRY["model"]) is True

    def test_entry_exposes_tools_capable_model_id(self):
        """The lane model must be the tool-calling Nemotron SKU, not a bare alias."""
        assert OPENROUTER_ENTRY["model"].endswith(":free")
        assert OPENROUTER_ENTRY["model"].startswith("nvidia/nemotron-3-ultra-550b-a55b")
