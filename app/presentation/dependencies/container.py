"""Dependency injection container for FastAPI.

All infrastructure and application layer objects are created here and
injected via FastAPI's Depends mechanism.  Components are created once
at application startup (lifespan) and shared across requests.

The pattern used is module-level singletons that are populated during
the application lifespan, then accessed via Depends() provider functions.
This avoids the caching issues of @lru_cache on async-incompatible code
and allows clean teardown.
"""

from __future__ import annotations

from app.application.services.async_extraction_service import AsyncExtractionService
from app.application.services.chunking_service import ChunkingService
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.prompt_builder import build_extraction_prompt  # noqa: F401 — re-exported
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.infrastructure.pdf.document_reader import DocumentReader
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore

# ── Module-level singletons ────────────────────────────────────────────────────
# Populated by initialise() during FastAPI lifespan startup.

_settings = None
_database: SQLiteDatabase | None = None
_document_repository: SQLiteDocumentRepository | None = None
_document_reader: DocumentReader | None = None
_embedding_client: OllamaEmbeddingClient | None = None
_llm_client: OllamaLLMClient | None = None
_vector_store: ChromaVectorStore | None = None
_chunking_service: ChunkingService | None = None
_text_cleaner: TextCleaner | None = None
_metadata_validator: MetadataValidator | None = None
_extract_use_case: ExtractMetadataUseCase | None = None
_async_extraction_service: AsyncExtractionService | None = None


def initialise() -> None:
    """Build the full dependency graph. Called once at application startup.
    
    Note: Ollama clients are created without connectivity checks during init.
    They will validate connectivity on first use, allowing the app to start
    even if Ollama is momentarily unavailable.
    """
    global _settings, _database, _document_repository, _document_reader
    global _embedding_client, _llm_client, _vector_store
    global _chunking_service, _text_cleaner, _metadata_validator, _extract_use_case
    global _async_extraction_service

    _settings = get_settings()

    # Infrastructure — database and document reading are immediate
    _database = SQLiteDatabase(_settings)
    _database.initialise()

    _document_repository = SQLiteDocumentRepository(_database)
    _document_reader = DocumentReader()
    
    # Ollama clients — created without connectivity checks (lazy validation on first call)
    _embedding_client = OllamaEmbeddingClient(_settings)
    _llm_client = OllamaLLMClient(_settings)
    _vector_store = ChromaVectorStore(_settings, _embedding_client)

    # Application services
    _text_cleaner = TextCleaner()
    _chunking_service = ChunkingService(_settings)
    _metadata_validator = MetadataValidator()
    _extract_use_case = ExtractMetadataUseCase(
        document_repository=_document_repository,
        document_reader=_document_reader,
        text_cleaner=_text_cleaner,
        chunking_service=_chunking_service,
        vector_store=_vector_store,
        llm_client=_llm_client,
        metadata_validator=_metadata_validator,
        settings=_settings,
        retrieval_k=_settings.retrieval_k,
    )
    
    # Async extraction service for parallel processing
    _async_extraction_service = AsyncExtractionService(
        use_case=_extract_use_case,
        settings=_settings,
        max_concurrent=2,
    )


# ── FastAPI Depends provider functions ────────────────────────────────────────

def get_extract_use_case() -> ExtractMetadataUseCase:
    assert _extract_use_case is not None, "Container not initialised"
    return _extract_use_case


def get_document_repository() -> DocumentRepository:
    assert _document_repository is not None, "Container not initialised"
    return _document_repository


def get_async_extraction_service() -> AsyncExtractionService:
    assert _async_extraction_service is not None, "Container not initialised"
    return _async_extraction_service
