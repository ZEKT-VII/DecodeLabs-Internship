"""
Memory Exam: Deterministic mocked tests + optional live API memory retention tests.

The DecodeLabs spec requires a "Memory Exam" where the chatbot must remember
user-provided facts across turns. We implement this as:

1. Mocked tests (default, fast, deterministic, no API key needed)
2. Live tests (marked @pytest.mark.live, requires NVIDIA_API_KEY)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from memory import MemoryManager
from schemas import MessageRole, ChatResponse, TokenUsage
from engine import ConversationEngine


# ════════════════════════════════════════════════════════════
# SECTION 1: Mocked Deterministic Memory Exam
# ════════════════════════════════════════════════════════════

class TestMockedMemoryExam:
    """
    Simulates multi-turn conversations with mocked LLM responses
    to verify that the memory architecture correctly retains and
    recalls user-provided facts.
    """

    def test_name_retention_in_context(self):
        """
        Simulates: User says their name → Verifies name appears in subsequent context.
        """
        mm = MemoryManager(system_prompt="You are helpful.")
        # Turn 1: User introduces themselves
        mm.add_message(MessageRole.USER, "My name is Alice.")
        mm.add_message(MessageRole.ASSISTANT, "Nice to meet you, Alice!")

        # Turn 2: User asks something else
        mm.add_message(MessageRole.USER, "What's the weather like?")

        # Verify: "Alice" still present in the assembled context
        context = mm.build_context()
        full_text = " ".join(m["content"] for m in context)
        assert "Alice" in full_text

    def test_pinned_identity_persists_after_window_pruning(self):
        """
        Pins user name to P2 → fills window to force pruning → checks name survives.
        """
        mm = MemoryManager(
            system_prompt="Short",
            max_context_tokens=200,
            reserved_output_tokens=50,
        )
        # Pin user identity at P2
        mm.set_identity_fact("name", "Bob")

        # Fill window with many messages to trigger pruning
        for i in range(15):
            mm.add_message(MessageRole.USER, f"Some padding message number {i} with extra content")
            mm.add_message(MessageRole.ASSISTANT, f"Reply to padding message {i} with filler")

        # Verify: "Bob" survives in the system prompt even after pruning
        context = mm.build_context()
        assert "Bob" in context[0]["content"]
        assert mm.get_stats()["pruned_messages"] > 0

    def test_preference_persists_across_turns(self):
        """
        User /remembers a preference → preference appears in subsequent contexts.
        """
        mm = MemoryManager()
        mm.add_preference("favorite_language", "Python")
        mm.add_message(MessageRole.USER, "Tell me a joke.")
        mm.add_message(MessageRole.ASSISTANT, "Why do programmers prefer dark mode?")
        mm.add_message(MessageRole.USER, "What's my favorite language?")

        context = mm.build_context()
        system_content = context[0]["content"]
        assert "Python" in system_content

    def test_multiple_facts_retained(self):
        """
        User provides multiple facts → all retained in context.
        """
        mm = MemoryManager()
        mm.set_identity_fact("name", "Charlie")
        mm.set_identity_fact("role", "Engineer")
        mm.add_preference("editor", "VSCode")
        mm.add_preference("os", "Linux")

        mm.add_message(MessageRole.USER, "Tell me about myself")

        context = mm.build_context()
        system_text = context[0]["content"]
        assert "Charlie" in system_text
        assert "Engineer" in system_text
        assert "VSCode" in system_text
        assert "Linux" in system_text

    def test_conversation_continuity_after_clear(self):
        """
        After /clear, pinned facts remain but rolling window is empty.
        """
        mm = MemoryManager()
        mm.set_identity_fact("name", "Diana")
        mm.add_message(MessageRole.USER, "Hello")
        mm.add_message(MessageRole.ASSISTANT, "Hi Diana!")

        mm.clear_rolling_window()

        context = mm.build_context()
        # Pinned identity should survive
        assert "Diana" in context[0]["content"]
        # Rolling window should be cleared
        assert len(context) == 1  # Only system message


class TestMockedEngineMemoryExam:
    """
    Tests the full engine pipeline with mocked LLM responses to verify
    end-to-end memory persistence across turns.
    """

    @patch("engine.LLMClient")
    @patch("engine.SessionStore")
    def test_engine_remembers_across_turns(self, MockStore, MockLLM):
        """
        Mocks the LLM to echo back context, verifying that the engine
        properly maintains memory state across multiple send_message calls.
        """
        # Configure mocked store
        mock_store = MockStore.return_value
        mock_store.create_session.return_value = "test-session-123"
        mock_store.save_message.return_value = "msg-id"

        # Configure mocked LLM
        mock_llm = MockLLM.return_value

        def mock_chat(messages, **kwargs):
            return ChatResponse(
                content="I remember what you said.",
                usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
                model="test-model",
                latency_sec=0.1,
            )

        mock_llm.chat.side_effect = mock_chat
        mock_llm.get_usage_stats.return_value = {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 40,
            "total_tokens": 140,
            "total_requests": 2,
        }

        # Create engine with mocked dependencies
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.model = "test-model"
        engine.memory = MemoryManager(system_prompt="You are helpful.")
        engine.persistence = mock_store
        engine.llm = mock_llm
        engine.session_id = "test-session-123"
        engine._created_at = None

        # Turn 1: Provide a fact
        engine.send_message("My name is Eve.")

        # Turn 2: Ask about the fact
        engine.send_message("What is my name?")

        # Verify: "Eve" appears in the context sent to the LLM
        last_call = mock_llm.chat.call_args_list[-1]
        # messages may be passed as positional or keyword arg
        if last_call.args:
            last_call_messages = last_call.args[0]
        else:
            last_call_messages = last_call.kwargs.get("messages", [])
        full_text = " ".join(m["content"] for m in last_call_messages)
        assert "Eve" in full_text


# ════════════════════════════════════════════════════════════
# SECTION 2: Live Memory Exam (requires NVIDIA_API_KEY)
# ════════════════════════════════════════════════════════════

@pytest.mark.live
class TestLiveMemoryExam:
    """
    Live tests that verify actual LLM memory retention with the NVIDIA endpoint.
    Run with: pytest tests/test_memory_exam.py -m live
    """

    @pytest.fixture
    def live_engine(self):
        """Creates a real ConversationEngine for live testing."""
        try:
            engine = ConversationEngine()
            engine.start_session()
            yield engine
            engine.shutdown()
        except Exception as e:
            pytest.skip(f"Live API not available: {e}")

    def test_live_name_recall(self, live_engine):
        """
        Live test: Tell the bot your name, then ask it back.
        Uses fuzzy matching since LLM responses vary.
        """
        # Turn 1: Introduce yourself
        live_engine.send_message("My name is Zephyr.")

        # Turn 2: Ask for recall
        response = live_engine.send_message("What is my name?")

        # Fuzzy assertion: "Zephyr" should appear in the response
        assert "zephyr" in response.content.lower(), (
            f"Expected 'Zephyr' in response, got: {response.content}"
        )

    def test_live_multi_fact_recall(self, live_engine):
        """
        Live test: Provide multiple facts, ask for recall.
        """
        live_engine.send_message("I am 25 years old and I live in Tokyo.")
        live_engine.send_message("I work as a data scientist.")

        response = live_engine.send_message(
            "Can you tell me what you know about me? My age, city, and job?"
        )

        content_lower = response.content.lower()
        assert "25" in content_lower or "twenty" in content_lower
        assert "tokyo" in content_lower
        assert "data scientist" in content_lower or "data" in content_lower
