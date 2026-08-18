"""ChatMessage entity representing a conversational interaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChatMessage:
    """Domain representation of a chat interaction."""

    id: str
    conversation_id: str
    document_id: str  # Document ID or "all"
    user_query: str
    assistant_response: str
    grounded: bool = True
    sources_json: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
