"""Unit tests for the MemoryManager with priority-tiered pinning."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from memory import MemoryManager
from schemas import MessageRole


class TestMemoryManagerBasic:
    """Tests for basic memory operations."""

    def test_initial_context_has_system_prompt(self):
        mm = MemoryManager(system_prompt="Test system prompt")
        context = mm.build_context()
        assert len(context) == 1
        assert context[0]["role"] == "system"
        assert "Test system prompt" in context[0]["content"]

    def test_add_user_message(self):
        mm = MemoryManager()
        mm.add_message(MessageRole.USER, "Hello")
        context = mm.build_context()
        assert len(context) == 2
        assert context[1]["role"] == "user"
        assert context[1]["content"] == "Hello"

    def test_add_assistant_message(self):
        mm = MemoryManager()
        mm.add_message(MessageRole.USER, "Hello")
        mm.add_message(MessageRole.ASSISTANT, "Hi there!")
        context = mm.build_context()
        assert len(context) == 3
        assert context[2]["role"] == "assistant"

    def test_clear_rolling_window(self):
        mm = MemoryManager()
        mm.add_message(MessageRole.USER, "Turn 1")
        mm.add_message(MessageRole.ASSISTANT, "Response 1")
        mm.clear_rolling_window()
        context = mm.build_context()
        assert len(context) == 1  # Only system prompt remains


class TestPinnedFacts:
    """Tests for P2 identity and P3 preference pinning."""

    def test_pin_identity_fact(self):
        mm = MemoryManager()
        mm.set_identity_fact("name", "Alice")
        context = mm.build_context()
        assert "[USER IDENTITY]" in context[0]["content"]
        assert "Alice" in context[0]["content"]

    def test_pin_preference(self):
        mm = MemoryManager()
        mm.add_preference("language", "Python")
        context = mm.build_context()
        assert "[USER PREFERENCES]" in context[0]["content"]
        assert "Python" in context[0]["content"]

    def test_pinned_facts_survive_clear(self):
        mm = MemoryManager()
        mm.set_identity_fact("name", "Bob")
        mm.add_preference("tone", "formal")
        mm.add_message(MessageRole.USER, "Hello")
        mm.clear_rolling_window()
        context = mm.build_context()
        # Pinned facts should remain in system message
        assert "Bob" in context[0]["content"]
        assert "formal" in context[0]["content"]
        # But rolling window should be empty
        assert len(context) == 1

    def test_remove_preference(self):
        mm = MemoryManager()
        mm.add_preference("color", "blue")
        assert mm.remove_preference("color") is True
        assert mm.remove_preference("nonexistent") is False


class TestTokenBudgetPruning:
    """Tests for automatic P4 pruning under token budget constraints."""

    def test_pruning_occurs_on_overflow(self):
        # Use a very tight budget to force pruning
        mm = MemoryManager(
            system_prompt="Short system prompt",
            max_context_tokens=200,
            reserved_output_tokens=50,
        )
        # Add many messages to exceed budget
        for i in range(20):
            mm.add_message(MessageRole.USER, f"Message {i} with some extra content padding")
            mm.add_message(MessageRole.ASSISTANT, f"Response {i} with some extra padding too")

        stats = mm.get_stats()
        assert stats["pruned_messages"] > 0
        # Context should fit within budget
        assert stats["estimated_context_tokens"] <= stats["budget_tokens"]

    def test_oldest_messages_pruned_first(self):
        mm = MemoryManager(
            system_prompt="Short",
            max_context_tokens=300,
            reserved_output_tokens=50,
        )
        mm.add_message(MessageRole.USER, "First message")
        mm.add_message(MessageRole.ASSISTANT, "First reply")
        mm.add_message(MessageRole.USER, "Second message with more content")
        mm.add_message(MessageRole.ASSISTANT, "Second reply with more content")
        mm.add_message(MessageRole.USER, "Third message with even more content filler")
        mm.add_message(MessageRole.ASSISTANT, "Third reply with even more content filler")

        context = mm.build_context()
        # Most recent messages should be present
        content_text = " ".join(m["content"] for m in context)
        assert "Third" in content_text


class TestMemoryStats:
    """Tests for statistics reporting."""

    def test_stats_structure(self):
        mm = MemoryManager()
        mm.add_message(MessageRole.USER, "Test")
        stats = mm.get_stats()
        assert "active_turns" in stats
        assert "pinned_identity_facts" in stats
        assert "pinned_preferences" in stats
        assert "pruned_messages" in stats
        assert "estimated_context_tokens" in stats
        assert "budget_tokens" in stats
        assert stats["active_turns"] == 1

    def test_reset_clears_all(self):
        mm = MemoryManager()
        mm.set_identity_fact("name", "Test")
        mm.add_preference("lang", "Python")
        mm.add_message(MessageRole.USER, "Hello")
        mm.reset()
        stats = mm.get_stats()
        assert stats["active_turns"] == 0
        assert stats["pinned_identity_facts"] == 0
        assert stats["pinned_preferences"] == 0
