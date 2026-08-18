"""Abstract repository interface for chat history."""

from abc import ABC, abstractmethod
from app.domain.entities.chat_message import ChatMessage


class ChatRepository(ABC):
    """Contract for persisting and retrieving chat messages."""

    @abstractmethod
    def save(self, message: ChatMessage) -> None:
        """Persist a new chat message."""

    @abstractmethod
    def find_by_document_id(self, document_id: str) -> list[ChatMessage]:
        """Fetch all messages for a specific document or 'all' scope."""

    @abstractmethod
    def find_by_conversation_id(self, conversation_id: str) -> list[ChatMessage]:
        """Fetch all messages belonging to a conversation."""
