"""SQLite implementation of the ChatRepository interface."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.domain.entities.chat_message import ChatMessage
from app.domain.exceptions.base import DatabaseException
from app.domain.repositories.chat_repository import ChatRepository
from app.infrastructure.database.sqlite_database import SQLiteDatabase

logger = logging.getLogger(__name__)


class SQLiteChatRepository(ChatRepository):
    """Persists ChatMessage aggregates in SQLite database."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._db = database

    def save(self, message: ChatMessage) -> None:
        sql = """
            INSERT INTO chat_history
                (id, conversation_id, document_id, user_query, assistant_response,
                 grounded, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            message.id,
            message.conversation_id,
            message.document_id,
            message.user_query,
            message.assistant_response,
            1 if message.grounded else 0,
            message.sources_json,
            message.created_at.isoformat(),
        )
        try:
            with self._db.connection() as conn:
                conn.execute(sql, params)
            logger.debug("Saved chat message id=%s", message.id)
        except Exception as exc:
            raise DatabaseException(f"Failed to save chat message {message.id}: {exc}") from exc

    def find_by_document_id(self, document_id: str) -> list[ChatMessage]:
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM chat_history WHERE document_id = ? ORDER BY created_at ASC",
                    (document_id,)
                ).fetchall()
            return [self._to_entity(dict(r)) for r in rows]
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch chat history for document {document_id}: {exc}") from exc

    def find_by_conversation_id(self, conversation_id: str) -> list[ChatMessage]:
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM chat_history WHERE conversation_id = ? ORDER BY created_at ASC",
                    (conversation_id,)
                ).fetchall()
            return [self._to_entity(dict(r)) for r in rows]
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch chat history for conversation {conversation_id}: {exc}") from exc

    def _to_entity(self, row: dict) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            document_id=row["document_id"],
            user_query=row["user_query"],
            assistant_response=row["assistant_response"],
            grounded=bool(row["grounded"]),
            sources_json=row.get("sources_json"),
            created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
        )
