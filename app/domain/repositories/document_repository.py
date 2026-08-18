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
    def find_by_hash(self, file_hash: str) -> Document | None:
        """Return the completed Document with the matching SHA-256 hash, or None."""

    @abstractmethod
    def find_all(self) -> list[Document]:
        """Return all stored documents ordered by creation date descending (includes duplicate upload history)."""

    @abstractmethod
    def find_all_canonical(self) -> list[Document]:
        """Return unique canonical documents (deduplicated by file_hash) for chat/search operations."""

    @abstractmethod
    def find_latest_by_filename(self, filename: str) -> Document | None:
        """Return the latest Document matching the given filename, or None."""

    @abstractmethod
    def deactivate_previous_versions(self, filename: str, current_id: str) -> None:
        """Set is_active=0 for older versions of the logical document identified by filename."""

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Remove a document record. Raises DocumentNotFoundException if not found."""
