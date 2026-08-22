"""
Hybrid Memory Manager with Prioritized Pinning & Token-Budgeted Rolling Window.

Memory Priority Tiers:
    P1 (System Prompt)      — Always present, immovable.
    P2 (User Identity)      — Pinned identity facts (name, role, etc.).
    P3 (User Preferences)   — Pinned via /remember command.
    P4 (Rolling Window)     — Token-budgeted sliding window of conversation turns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_RESERVED_OUTPUT_TOKENS
from schemas import Message, MessageRole
from token_budget import (
    estimate_message_tokens,
    compute_context_budget,
)

logger = logging.getLogger("stateful_chatbot.memory")


class MemoryManager:
    """
    Manages conversation context as a prioritized, token-budgeted message list.
    
    Guarantees:
    - P1-P3 (pinned) layers are always present in every API call.
    - P4 (rolling window) is pruned oldest-first when headroom is exhausted.
    - Total estimated tokens never exceed budget.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
    ) -> None:
        self._max_context = max_context_tokens
        self._reserved_output = reserved_output_tokens

        # P1: System Prompt (immovable)
        self._system_message = Message(role=MessageRole.SYSTEM, content=system_prompt)

        # Global Memory Facts (shared across all sessions)
        self._global_facts: Dict[str, str] = {}

        # P2: User Identity Facts (pinned per session)
        self._identity_facts: Dict[str, str] = {}

        # P3: User Preferences (pinned via /remember per session)
        self._preferences: Dict[str, str] = {}

        # P4: Rolling Conversation Window
        self._rolling_window: List[Message] = []

        # Statistics
        self._pruned_count: int = 0

    # ──────────────────────────── Global Memory ──────────────────────────

    def set_global_facts(self, facts: Dict[str, str]) -> None:
        """Sets the global facts shared across all conversations."""
        self._global_facts = dict(facts)

    def get_all_global_facts(self) -> Dict[str, str]:
        return dict(self._global_facts)

    # ──────────────────────────── P2: Identity ──────────────────────────

    def set_identity_fact(self, key: str, value: str) -> None:
        """Pins a user identity fact (e.g., 'name', 'role')."""
        self._identity_facts[key.lower().strip()] = value.strip()
        logger.info("Pinned identity fact: %s", key)

    def get_identity_fact(self, key: str) -> Optional[str]:
        """Retrieves a pinned identity fact."""
        return self._identity_facts.get(key.lower().strip())

    def get_all_identity_facts(self) -> Dict[str, str]:
        return dict(self._identity_facts)

    # ──────────────────────────── P3: Preferences ──────────────────────────

    def add_preference(self, key: str, value: str) -> None:
        """Pins a user preference via /remember."""
        self._preferences[key.lower().strip()] = value.strip()
        logger.info("Pinned preference: %s", key)

    def remove_preference(self, key: str) -> bool:
        """Removes a pinned preference."""
        return self._preferences.pop(key.lower().strip(), None) is not None

    def get_all_preferences(self) -> Dict[str, str]:
        return dict(self._preferences)

    # ──────────────────────────── P4: Rolling Window ──────────────────────────

    def add_message(self, role: MessageRole, content: str) -> None:
        """Adds a new conversational turn to the rolling window and auto-prunes if needed."""
        msg = Message(role=role, content=content)
        self._rolling_window.append(msg)
        self._prune_to_budget()

    def clear_rolling_window(self) -> None:
        """Clears the P4 rolling window (preserves pinned facts)."""
        self._rolling_window.clear()
        logger.info("Rolling window cleared.")

    # ──────────────────────────── Context Assembly ──────────────────────────

    def build_context(self) -> List[Dict[str, str]]:
        """
        Assembles the complete message payload for the API request.
        
        Priority order:
            P1: System prompt
            Global: Global user memory (injected into system prompt)
            P2: Identity facts (injected into system prompt addendum)
            P3: User preferences (injected into system prompt addendum)
            P4: Rolling conversation window
        """
        # Build the combined system message
        system_content = self._system_message.content

        # Append Global Memory Facts (shared)
        if self._global_facts:
            global_str = "; ".join(f"{k}: {v}" for k, v in self._global_facts.items())
            system_content += f"\n\n[GLOBAL USER CONTEXT / MEMORY] {global_str}"

        # Append P2 identity facts
        if self._identity_facts:
            identity_str = "; ".join(f"{k}: {v}" for k, v in self._identity_facts.items())
            system_content += f"\n\n[USER IDENTITY] {identity_str}"

        # Append P3 preferences
        if self._preferences:
            pref_str = "; ".join(f"{k}: {v}" for k, v in self._preferences.items())
            system_content += f"\n\n[USER PREFERENCES] {pref_str}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

        # P4: Rolling conversation turns
        for msg in self._rolling_window:
            messages.append(msg.to_api_dict())

        return messages

    # ──────────────────────────── Budget Management ──────────────────────────

    def _prune_to_budget(self) -> None:
        """
        Prunes the oldest P4 (rolling window) turns until the assembled context
        fits within the computed token budget.
        """
        budget = compute_context_budget(self._max_context, self._reserved_output)

        while self._rolling_window:
            context = self.build_context()
            total_tokens = sum(estimate_message_tokens(m) for m in context)

            if total_tokens <= budget:
                break

            # Prune the oldest conversation turn from P4
            removed = self._rolling_window.pop(0)
            self._pruned_count += 1
            logger.debug(
                "Pruned oldest message (role=%s, est_tokens=%d). Total pruned: %d",
                removed.role.value,
                estimate_message_tokens(removed.to_api_dict()),
                self._pruned_count,
            )

    def get_estimated_tokens(self) -> int:
        """Returns the current estimated token usage for the full context."""
        context = self.build_context()
        return sum(estimate_message_tokens(m) for m in context)

    # ──────────────────────────── Statistics ──────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Returns memory management statistics."""
        return {
            "active_turns": len(self._rolling_window),
            "global_facts_count": len(self._global_facts),
            "pinned_identity_facts": len(self._identity_facts),
            "pinned_preferences": len(self._preferences),
            "pruned_messages": self._pruned_count,
            "estimated_context_tokens": self.get_estimated_tokens(),
            "budget_tokens": compute_context_budget(self._max_context, self._reserved_output),
        }

    def reset(self) -> None:
        """Full reset: clears all tiers except P1 system prompt."""
        self._identity_facts.clear()
        self._preferences.clear()
        self._rolling_window.clear()
        self._pruned_count = 0
        logger.info("Memory fully reset (P2-P4 cleared).")
