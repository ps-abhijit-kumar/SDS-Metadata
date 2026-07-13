"""Abstract document repository interface.

The domain defines what persistence operations it needs.
Infrastructure provides the concrete implementation.
This ensures the domain never depends on SQLite, Postgres, or any other
storage technology directly.
"""

from abc import ABC, abstractmethod

from app.domain.entities.document import Document


class DocumentRepository(ABC):
    """Contract for persisting and retrieving Document aggregates."""

    @abstractmethod
    def save(self, document: Document) -> None:
        """Persist a new document record. Raises DuplicateDocumentException if ID exists."""

    @abstractmethod
    def update(self, document: Document) -> None:
        """Update an existing document record. Raises DocumentNotFoundException if not found."""

    @abstractmethod
    def find_by_id(self, document_id: str) -> Document | None:
        """Return the Document with the given ID, or None if it does not exist."""

    @abstractmethod
    def find_all(self) -> list[Document]:
        """Return all stored documents ordered by creation date descending."""

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Remove a document record. Raises DocumentNotFoundException if not found."""
