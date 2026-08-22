"""Integration tests for ConversationEngine with mocked LLM client."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from engine import ConversationEngine
from memory import MemoryManager
from schemas import ChatResponse, TokenUsage, MessageRole
from exceptions import ContextOverflowError, APIRateLimitError, EmptyInputError


@pytest.fixture
def mocked_engine(tmp_path):
    """Creates a ConversationEngine with temporary SQLite DB and mocked LLM."""
    db_file = tmp_path / "test_engine.db"
    engine = ConversationEngine.__new__(ConversationEngine)
    engine.model = "deepseek-ai/deepseek-v4-flash-0731"
    engine._api_key = "test-key"
    engine.memory = MemoryManager(system_prompt="You are a helpful assistant.")
    
    # Real persistence using temporary SQLite
    from persistence import SessionStore
    engine.persistence = SessionStore(db_path=db_file)
    
    # Mock LLM Client
    mock_llm = MagicMock()
    mock_llm.chat.return_value = ChatResponse(
        content="Mocked assistant reply.",
        usage=TokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
        model="deepseek-ai/deepseek-v4-flash-0731",
        latency_sec=0.2,
    )
    mock_llm.get_usage_stats.return_value = {
        "total_prompt_tokens": 40,
        "total_completion_tokens": 15,
        "total_tokens": 55,
        "total_requests": 1,
    }
    engine.llm = mock_llm
    engine.session_id = None
    engine._created_at = None

    yield engine
    engine.shutdown()


class TestEngineLifecycle:
    def test_start_session_creates_db_record(self, mocked_engine):
        session_id = mocked_engine.start_session()
        assert session_id is not None
        sessions = mocked_engine.persistence.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id

    def test_send_message_auto_starts_session_if_none(self, mocked_engine):
        assert mocked_engine.session_id is None
        mocked_engine.send_message("Hello there!")
        assert mocked_engine.session_id is not None
        messages = mocked_engine.get_history()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


class TestEngineMultiTurn:
    def test_multi_turn_persists_and_accumulates(self, mocked_engine):
        mocked_engine.start_session()
        r1 = mocked_engine.send_message("Turn 1: My name is Alice")
        r2 = mocked_engine.send_message("Turn 2: What is my name?")
        
        history = mocked_engine.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "Turn 1: My name is Alice"
        assert history[1]["content"] == "Mocked assistant reply."
        assert history[2]["content"] == "Turn 2: What is my name?"
        assert history[3]["content"] == "Mocked assistant reply."

    def test_pinned_facts_injected_into_llm_call(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.set_user_identity("name", "Vipin")
        mocked_engine.remember_fact("favorite_language", "Python")

        mocked_engine.send_message("Tell me what you know")
        
        # Check messages passed to mocked LLM
        call = mocked_engine.llm.chat.call_args
        if call.args:
            sent_messages = call.args[0]
        else:
            sent_messages = call.kwargs.get("messages", [])
        system_content = sent_messages[0]["content"]

        assert "[USER IDENTITY] name: Vipin" in system_content
        assert "[USER PREFERENCES] favorite_language: Python" in system_content

    def test_forget_fact_removes_preference(self, mocked_engine):
        mocked_engine.remember_fact("editor", "Neovim")
        assert mocked_engine.forget_fact("editor") is True
        assert mocked_engine.forget_fact("nonexistent") is False

    def test_global_facts_injected_into_llm_call(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.set_global_fact("global_role", "Data Scientist")

        mocked_engine.send_message("Who am I?")
        
        call = mocked_engine.llm.chat.call_args
        sent_messages = call.args[0] if call.args else call.kwargs.get("messages", [])
        system_content = sent_messages[0]["content"]

        assert "[GLOBAL USER CONTEXT / MEMORY] global_role: Data Scientist" in system_content


class TestEngineErrorHandling:
    def test_empty_input_rejected_before_llm(self, mocked_engine):
        mocked_engine.start_session()
        with pytest.raises(EmptyInputError):
            mocked_engine.send_message("   ")
        mocked_engine.llm.chat.assert_not_called()

    def test_llm_failure_rolls_back_user_message_from_memory(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.llm.chat.side_effect = APIRateLimitError("Rate limit hit")

        with pytest.raises(APIRateLimitError):
            mocked_engine.send_message("This will fail at LLM layer")

        # Memory rolling window should be empty because failure rolled it back
        assert len(mocked_engine.memory._rolling_window) == 0


class TestEngineSessionOperations:
    def test_clear_memory_preserves_sqlite(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.send_message("Hello")
        assert len(mocked_engine.memory._rolling_window) == 2

        mocked_engine.clear_memory()
        assert len(mocked_engine.memory._rolling_window) == 0
        assert len(mocked_engine.get_history()) == 2

    def test_delete_session_removes_sqlite_and_resets_memory(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.send_message("Hello")
        sid = mocked_engine.session_id

        deleted = mocked_engine.delete_session()
        assert deleted is True
        assert mocked_engine.session_id is None
        assert len(mocked_engine.persistence.list_sessions()) == 0

    def test_get_stats_aggregates_memory_and_llm(self, mocked_engine):
        mocked_engine.start_session()
        mocked_engine.send_message("Hello")
        stats = mocked_engine.get_stats()
        assert stats["session_id"] == mocked_engine.session_id
        assert "total_prompt_tokens" in stats
        assert "estimated_context_tokens" in stats
