"""Unit tests for slash commands and router isolation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from commands import handle_command, COMMAND_REGISTRY
from exceptions import InvalidCommandError


@pytest.fixture
def mock_engine():
    """Provides a mocked engine instance for command tests."""
    engine = MagicMock()
    engine.session_id = "test-session-uuid-1234"
    engine.model = "deepseek-ai/deepseek-v4-flash-0731"
    engine.get_stats.return_value = {
        "session_id": "test-session-uuid-1234",
        "model": "deepseek-ai/deepseek-v4-flash-0731",
        "active_turns": 2,
        "pinned_identity_facts": 1,
        "pinned_preferences": 1,
        "pruned_messages": 0,
        "estimated_context_tokens": 150,
        "budget_tokens": 7000,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_requests": 2,
    }
    engine.get_history.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    engine.delete_session.return_value = True
    engine.delete_all_sessions.return_value = 3
    engine.forget_fact.return_value = True
    return engine


class TestCommandHelp:
    def test_help_lists_all_registry_commands(self, mock_engine):
        response, should_exit = handle_command("help", "", mock_engine)
        assert should_exit is False
        for cmd in COMMAND_REGISTRY:
            assert f"/{cmd}" in response


class TestCommandStats:
    def test_stats_formatting(self, mock_engine):
        response, should_exit = handle_command("stats", "", mock_engine)
        assert should_exit is False
        assert "Session ID" in response
        assert "test-session-uuid-1234" in response
        assert "Context Tokens" in response
        assert "API Key:" in response


class TestCommandHistory:
    def test_history_with_messages(self, mock_engine):
        response, should_exit = handle_command("history", "", mock_engine)
        assert should_exit is False
        assert "USER: Hello" in response
        assert "ASSISTANT: Hi there!" in response

    def test_history_empty(self, mock_engine):
        mock_engine.get_history.return_value = []
        response, should_exit = handle_command("history", "", mock_engine)
        assert "No conversation history yet" in response


class TestCommandClear:
    def test_clear_calls_engine_clear_memory(self, mock_engine):
        response, should_exit = handle_command("clear", "", mock_engine)
        assert should_exit is False
        mock_engine.clear_memory.assert_called_once()
        assert "cleared" in response.lower()


class TestCommandDelete:
    def test_delete_with_force_flag(self, mock_engine):
        response, should_exit = handle_command("delete", "--force", mock_engine)
        assert should_exit is False
        mock_engine.delete_session.assert_called_once()
        assert "deleted" in response.lower()

    def test_delete_without_force_cancelled(self, mock_engine):
        with patch("builtins.input", return_value="n"):
            response, should_exit = handle_command("delete", "", mock_engine)
            assert "cancelled" in response.lower()
            mock_engine.delete_session.assert_not_called()

    def test_delete_without_force_confirmed(self, mock_engine):
        with patch("builtins.input", return_value="y"):
            response, should_exit = handle_command("delete", "", mock_engine)
            assert "deleted" in response.lower()
            mock_engine.delete_session.assert_called_once()


class TestCommandDeleteAll:
    def test_delete_all_with_force_flag(self, mock_engine):
        response, should_exit = handle_command("delete-all", "--force", mock_engine)
        assert should_exit is False
        mock_engine.delete_all_sessions.assert_called_once()
        assert "Purged 3" in response

    def test_delete_all_without_force_cancelled(self, mock_engine):
        with patch("builtins.input", return_value="no"):
            response, should_exit = handle_command("delete-all", "", mock_engine)
            assert "cancelled" in response.lower()
            mock_engine.delete_all_sessions.assert_not_called()


class TestCommandRememberAndForget:
    def test_remember_valid_format(self, mock_engine):
        response, should_exit = handle_command("remember", "lang=Python", mock_engine)
        assert should_exit is False
        mock_engine.remember_fact.assert_called_once_with("lang", "Python")
        assert "Remembered: lang = Python" in response

    def test_remember_invalid_format(self, mock_engine):
        response, should_exit = handle_command("remember", "invalid_no_equals", mock_engine)
        assert "Usage: /remember key=value" in response

    def test_forget_existing_key(self, mock_engine):
        response, should_exit = handle_command("forget", "lang", mock_engine)
        assert should_exit is False
        mock_engine.forget_fact.assert_called_once_with("lang")
        assert "Forgotten: lang" in response

    def test_forget_nonexistent_key(self, mock_engine):
        mock_engine.forget_fact.return_value = False
        response, should_exit = handle_command("forget", "nonexistent", mock_engine)
        assert "was not found" in response

    def test_forget_empty_arg(self, mock_engine):
        response, should_exit = handle_command("forget", "", mock_engine)
        assert "Usage: /forget key" in response


class TestCommandKeyStatus:
    def test_key_status_output_is_masked(self, mock_engine):
        response, should_exit = handle_command("key-status", "", mock_engine)
        assert should_exit is False
        assert "API Key Status" in response
        assert "Source:" in response
        assert "Preview:" in response
        # Ensure raw API keys are never exposed in key-status
        assert "nvapi-dummytestkey00000000000000000000000000000000000000000000000000000" not in response


class TestCommandModel:
    def test_model_display(self, mock_engine):
        response, should_exit = handle_command("model", "", mock_engine)
        assert should_exit is False
        assert "deepseek-ai/deepseek-v4-flash-0731" in response


class TestCommandExit:
    def test_exit_returns_should_exit_true(self, mock_engine):
        response, should_exit = handle_command("exit", "", mock_engine)
        assert should_exit is True
        assert "Goodbye" in response

    def test_quit_returns_should_exit_true(self, mock_engine):
        response, should_exit = handle_command("quit", "", mock_engine)
        assert should_exit is True
        assert "Goodbye" in response


class TestUnknownCommand:
    def test_unknown_command_raises_invalid_command_error(self, mock_engine):
        with pytest.raises(InvalidCommandError) as exc_info:
            handle_command("invalid_unknown_cmd", "", mock_engine)
        assert "Unknown command: /invalid_unknown_cmd" in str(exc_info.value)
