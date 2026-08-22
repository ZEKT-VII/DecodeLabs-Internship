"""Unit tests for the SQLite persistence layer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import tempfile
from pathlib import Path
from persistence import SessionStore


@pytest.fixture
def store(tmp_path):
    """Creates a temporary SessionStore for testing."""
    db_path = tmp_path / "test_sessions.db"
    s = SessionStore(db_path=db_path)
    yield s
    s.close()


class TestSessionOperations:
    """Tests for session lifecycle management."""

    def test_create_session(self, store):
        session_id = store.create_session(model="test-model")
        assert session_id is not None
        assert len(session_id) == 36  # UUID format

    def test_list_sessions(self, store):
        store.create_session(model="model-1")
        store.create_session(model="model-2")
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_delete_session(self, store):
        sid = store.create_session()
        deleted = store.delete_session(sid)
        assert deleted is True
        sessions = store.list_sessions()
        assert len(sessions) == 0

    def test_delete_nonexistent_session(self, store):
        deleted = store.delete_session("nonexistent-uuid")
        assert deleted is False

    def test_delete_all_sessions(self, store):
        store.create_session()
        store.create_session()
        store.create_session()
        count = store.delete_all_sessions()
        assert count == 3
        sessions = store.list_sessions()
        assert len(sessions) == 0


class TestMessageOperations:
    """Tests for message persistence."""

    def test_save_and_retrieve_messages(self, store):
        sid = store.create_session()
        store.save_message(sid, "user", "Hello", token_count=5)
        store.save_message(sid, "assistant", "Hi there!", token_count=8)

        messages = store.get_session_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_message_count(self, store):
        sid = store.create_session()
        store.save_message(sid, "user", "Message 1")
        store.save_message(sid, "assistant", "Reply 1")
        store.save_message(sid, "user", "Message 2")

        count = store.get_session_message_count(sid)
        assert count == 3

    def test_messages_ordered_chronologically(self, store):
        sid = store.create_session()
        for i in range(5):
            store.save_message(sid, "user", f"Turn {i}")

        messages = store.get_session_messages(sid)
        for i, msg in enumerate(messages):
            assert msg["content"] == f"Turn {i}"

    def test_cascade_delete_removes_messages(self, store):
        sid = store.create_session()
        store.save_message(sid, "user", "Test message")
        store.save_message(sid, "assistant", "Test reply")

        store.delete_session(sid)

        # Messages should be gone after session deletion
        messages = store.get_session_messages(sid)
        assert len(messages) == 0

    def test_message_limit(self, store):
        sid = store.create_session()
        for i in range(10):
            store.save_message(sid, "user", f"Msg {i}")

        limited = store.get_session_messages(sid, limit=3)
        assert len(limited) == 3


class TestDatabaseInit:
    """Tests for WAL mode and schema creation."""

    def test_wal_mode_enabled(self, store):
        conn = store._get_connection()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()
        assert mode[0].lower() == "wal"

    def test_foreign_keys_enabled(self, store):
        conn = store._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()
        assert fk[0] == 1


class TestGlobalMemoryOperations:
    """Tests for global memory persistence."""

    def test_set_and_get_global_facts(self, store):
        store.set_global_fact("language", "Python")
        store.set_global_fact("tone", "Concise")
        facts = store.get_all_global_facts()
        assert facts["language"] == "Python"
        assert facts["tone"] == "Concise"

    def test_update_global_fact(self, store):
        store.set_global_fact("role", "Developer")
        store.set_global_fact("role", "Lead Architect")
        facts = store.get_all_global_facts()
        assert facts["role"] == "Lead Architect"

    def test_delete_global_fact(self, store):
        store.set_global_fact("temp_key", "temp_value")
        deleted = store.delete_global_fact("temp_key")
        assert deleted is True
        facts = store.get_all_global_facts()
        assert "temp_key" not in facts
