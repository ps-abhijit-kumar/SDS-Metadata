"""Domain exception hierarchy.

All exceptions in this application inherit from ApplicationException so
callers can distinguish platform errors from unexpected Python exceptions
with a single except clause.
"""


class ApplicationException(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


# ── Domain ─────────────────────────────────────────────────────────────────────

class DocumentNotFoundException(ApplicationException):
    """Raised when a requested document does not exist in the repository."""


class DuplicateDocumentException(ApplicationException):
    """Raised when attempting to store a document with an ID that already exists."""


class InvalidDocumentException(ApplicationException):
    """Raised when the uploaded file is not a valid, processable document."""


# ── Infrastructure ─────────────────────────────────────────────────────────────

class PDFExtractionException(ApplicationException):
    """Raised when text cannot be extracted from a PDF file."""


class EmbeddingException(ApplicationException):
    """Raised when the embedding model fails to produce vectors."""


class VectorStoreException(ApplicationException):
    """Raised when the vector store (ChromaDB) encounters an error."""


class LLMException(ApplicationException):
    """Raised when the local LLM (Ollama) fails to generate a response."""


class DatabaseException(ApplicationException):
    """Raised when a SQLite database operation fails."""


# ── Application ────────────────────────────────────────────────────────────────

class MetadataExtractionException(ApplicationException):
    """Raised when metadata cannot be extracted or validated from LLM output."""


class ChunkingException(ApplicationException):
    """Raised when the document chunking pipeline fails."""


class ConfigurationException(ApplicationException):
    """Raised when required configuration is missing or invalid."""
