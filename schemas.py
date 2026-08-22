"""Data schemas and typed models for the Stateful Chatbot Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Supported roles in conversational payloads."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Structured representation of a single conversation turn."""
    role: MessageRole
    content: str
    token_count: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, str]:
        """Converts message into standard OpenAI/NVIDIA API role-content payload."""
        return {"role": self.role.value, "content": self.content}


class TokenUsage(BaseModel):
    """Token consumption accounting."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Structured response from the LLM engine."""
    content: str
    reasoning: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    latency_sec: float = 0.0


class SessionStats(BaseModel):
    """Telemetry and operational metrics for an active session."""
    session_id: str
    model: str
    total_turns: int = 0
    active_memory_turns: int = 0
    pinned_facts_count: int = 0
    estimated_context_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    pruned_messages_count: int = 0
    created_at: datetime
    updated_at: datetime
