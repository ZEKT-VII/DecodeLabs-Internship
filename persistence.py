"""SQLite WAL-mode persistence layer for session and message storage."""

from __future__ import annotations

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DEFAULT_DB_PATH, DEFAULT_MODEL
from exceptions import PersistenceError

logger = logging.getLogger("stateful_chatbot.persistence")


class SessionStore:
    """Thread-local SQLite storage for chat sessions and message history."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Lazily connects and configures the SQLite database."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
        return self._conn

    def _initialize_db(self) -> None:
        """Creates the schema if tables do not exist."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS global_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            """)
            conn.commit()
            logger.info("Database initialized at: %s", self.db_path)
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to initialize database: {e}")

    # ──────────────────────────── Session Operations ──────────────────────────

    def create_session(
        self,
        model: str = DEFAULT_MODEL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a new session record and returns its unique ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, model, created_at, updated_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, model, now, now, json.dumps(metadata or {})),
            )
            conn.commit()
            logger.info("Created session: %s", session_id)
            return session_id
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to create session: {e}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single session record by its ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT session_id, model, created_at, updated_at, metadata_json "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                res = dict(row)
                res["metadata"] = json.loads(res.pop("metadata_json", "{}"))
                return res
            return None
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to get session {session_id}: {e}")

    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """Updates the metadata_json field of an existing session."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE sessions SET metadata_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(metadata), now, session_id),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to update session metadata {session_id}: {e}")

    def get_session_pinned_facts(self, session_id: str) -> Dict[str, str]:
        """Retrieves session-specific pinned facts from metadata."""
        sess = self.get_session(session_id)
        if not sess:
            return {}
        return sess.get("metadata", {}).get("pinned_facts", {})

    def set_session_pinned_fact(self, session_id: str, key: str, value: str) -> None:
        """Sets or updates a session-specific pinned fact in persistence."""
        sess = self.get_session(session_id)
        if not sess:
            return
        meta = sess.get("metadata", {})
        pinned = meta.get("pinned_facts", {})
        pinned[key.strip().lower()] = value.strip()
        meta["pinned_facts"] = pinned
        self.update_session_metadata(session_id, meta)

    def delete_session_pinned_fact(self, session_id: str, key: str) -> bool:
        """Deletes a session-specific pinned fact from persistence."""
        sess = self.get_session(session_id)
        if not sess:
            return False
        meta = sess.get("metadata", {})
        pinned = meta.get("pinned_facts", {})
        cleaned_key = key.strip().lower()
        if cleaned_key in pinned:
            del pinned[cleaned_key]
            meta["pinned_facts"] = pinned
            self.update_session_metadata(session_id, meta)
            return True
        return False

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent sessions with message counts and first message snippets."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT 
                    s.session_id, 
                    s.model, 
                    s.created_at, 
                    s.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) as message_count,
                    (SELECT content FROM messages m WHERE m.session_id = s.session_id AND m.role = 'user' ORDER BY m.created_at ASC LIMIT 1) as title
                FROM sessions s
                ORDER BY s.updated_at DESC 
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to list sessions: {e}")

    def delete_session(self, session_id: str) -> bool:
        """Deletes a specific session and all its messages (cascading FK)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted session: %s", session_id)
            return deleted
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to delete session {session_id}: {e}")

    def delete_all_sessions(self) -> int:
        """Purges all sessions and messages from the database."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM sessions")
            conn.commit()
            count = cursor.rowcount
            logger.warning("Purged all sessions (%d records)", count)
            return count
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to purge all sessions: {e}")

    # ──────────────────────────── Message Operations ──────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persists a conversation message to the database."""
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO messages (message_id, session_id, role, content, token_count, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, token_count, now, json.dumps(metadata or {})),
            )
            # Update session's updated_at timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
            return message_id
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to save message: {e}")

    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves messages for a session, ordered chronologically."""
        conn = self._get_connection()
        try:
            query = (
                "SELECT role, content, token_count, created_at, metadata_json "
                "FROM messages WHERE session_id = ? ORDER BY created_at ASC"
            )
            params: list = [session_id]
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to load messages for session {session_id}: {e}")

    def get_session_message_count(self, session_id: str) -> int:
        """Returns the count of messages in a given session."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["cnt"] if row else 0
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to count messages: {e}")

    # ──────────────────────────── Global Memory Operations ──────────────────────────

    def set_global_fact(self, key: str, value: str) -> None:
        """Inserts or updates a global memory fact available across all sessions."""
        cleaned_key = key.strip().lower()
        cleaned_val = value.strip()
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO global_memory (key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (cleaned_key, cleaned_val, now, now),
            )
            conn.commit()
            logger.info("Saved global memory fact: %s", cleaned_key)
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to save global fact {key}: {e}")

    def get_all_global_facts(self) -> Dict[str, str]:
        """Retrieves all global memory facts as a key-value mapping."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT key, value FROM global_memory ORDER BY created_at ASC"
            ).fetchall()
            return {r["key"]: r["value"] for r in rows}
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to load global memory facts: {e}")

    def get_all_global_facts_detailed(self) -> List[Dict[str, Any]]:
        """Retrieves all global memory facts with timestamps for UI display."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT key, value, created_at, updated_at FROM global_memory ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to load global memory list: {e}")

    def delete_global_fact(self, key: str) -> bool:
        """Deletes a specific global memory fact."""
        cleaned_key = key.strip().lower()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM global_memory WHERE key = ?", (cleaned_key,)
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to delete global fact {key}: {e}")

    # ──────────────────────────── Lifecycle ──────────────────────────

    def close(self) -> None:
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")
