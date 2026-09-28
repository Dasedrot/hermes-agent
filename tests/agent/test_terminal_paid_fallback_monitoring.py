"""PART 3 — Focused monitoring tests for terminal-paid fallback WARNING.

A — terminal paid fallback: transition into deepseek/deepseek-flash produces exactly one terminal-paid WARNING.
B — NIM → NIM fallback: transition between NIM models produces zero terminal-paid WARNINGs.
C — repeated activations: two later terminal-paid fallback activations produce two WARNING events total.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _agent(provider="nvidia", model="nvidia/nemotron-3-ultra", is_subagent=False, platform=None):
    """Create a minimal agent-like namespace for testing."""
    return SimpleNamespace(
        provider=provider,
        model=model,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=None,
        _fallback_activated=False,
        _fallback_chain=[],
        _fallback_index=0,
        _rate_limited_until=0,
        _cascade_deadline=None,
        _unavailable_fallback_keys=set(),
        is_subagent=is_subagent,
        platform=platform,
        _config_context_length=None,
        requested_provider=provider,
        api_mode="chat_completions",
        _reasoning_echo_flag=False,
        _fallback_reason_text=lambda r: r.value if r else "unknown",
        _anthropic_prompt_cache_policy=lambda **kwargs: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
        runtime_capabilities=None,
    )


class TestTerminalPaidFallbackWarning:
    """PART 3 — Focused monitoring tests for terminal-paid fallback WARNING."""

    def test_A_terminal_paid_fallback_emits_warning(self, monkeypatch, caplog):
        """A — terminal paid fallback: transition into deepseek/deepseek-flash produces exactly one terminal-paid WARNING."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent()

        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.rate_limit,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-flash",
                fb_provider="deepseek",
            )

        # Filter for the specific terminal-paid WARNING
        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 1, f"Expected exactly 1 terminal-paid WARNING, got {len(terminal_paid_warnings)}"
        msg = terminal_paid_warnings[0].getMessage()
        assert "cascade reached terminal PAID fallback" in msg
        assert "primary=nvidia/nemotron-3-ultra" in msg
        assert "terminal=deepseek/deepseek-flash" in msg
        assert "reason=rate limit" in msg
        assert "role=architect" in msg  # default agent is not subagent

    def test_B_nim_to_nim_fallback_no_terminal_paid_warning(self, monkeypatch, caplog):
        """B — NIM → NIM fallback: transition between NIM models produces zero terminal-paid WARNINGs."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent()

        with caplog.at_level(logging.INFO, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.rate_limit,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="z-ai/glm-5.3",
                fb_provider="nvidia",
            )

        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 0, f"Expected 0 terminal-paid WARNINGs for NIM→NIM, got {len(terminal_paid_warnings)}"

        # Should still have INFO log for normal fallback
        info_logs = [r for r in caplog.records if r.levelno == logging.INFO and "Fallback activated" in r.getMessage()]
        assert len(info_logs) >= 1

    def test_C_repeated_terminal_paid_activations_produce_two_warnings(self, monkeypatch, caplog):
        """C — repeated activations: two later terminal-paid fallback activations produce two WARNING events total."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent()

        # First activation
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.rate_limit,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-flash",
                fb_provider="deepseek",
            )

        # Second activation (simulating later turn)
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.overloaded,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-flash",
                fb_provider="deepseek",
            )

        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 2, f"Expected 2 terminal-paid WARNINGs, got {len(terminal_paid_warnings)}"
        # First should have rate_limit reason
        assert "reason=rate limit" in terminal_paid_warnings[0].getMessage()
        # Second should have overloaded reason
        assert "reason=provider overloaded" in terminal_paid_warnings[1].getMessage()

    def test_terminal_paid_warning_with_billing_reason_emits_both(self, monkeypatch, caplog):
        """When terminal-paid fallback happens due to billing, both warnings should emit."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent()

        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.billing,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-flash",
                fb_provider="deepseek",
            )

        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]
        billing_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "refused for billing/credits" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 1, "Expected exactly 1 terminal-paid WARNING"
        assert len(billing_warnings) == 1, "Expected exactly 1 billing WARNING (preserved existing behavior)"

    def test_subagent_role_logged_as_delegated(self, monkeypatch, caplog):
        """Subagent should log role='delegated'."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent(is_subagent=True)

        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.rate_limit,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-flash",
                fb_provider="deepseek",
            )

        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 1
        assert "role=delegated" in terminal_paid_warnings[0].getMessage()

    def test_deepseek_v4_flash_not_terminal_paid(self, monkeypatch, caplog):
        """deepseek/deepseek-v4-flash (NIM) should NOT trigger terminal-paid WARNING."""
        from agent.chat_completion_helpers import _log_fallback_activated
        from agent.error_classifier import FailoverReason

        agent = _agent()

        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            _log_fallback_activated(
                agent,
                FailoverReason.rate_limit,
                old_model="nvidia/nemotron-3-ultra",
                old_provider="nvidia",
                fb_model="deepseek-v4-flash",
                fb_provider="deepseek",
            )

        terminal_paid_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "cascade reached terminal PAID fallback" in r.getMessage()
        ]

        assert len(terminal_paid_warnings) == 0, "deepseek-v4-flash is NIM, not terminal paid"