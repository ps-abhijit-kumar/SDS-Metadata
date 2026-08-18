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
from app.application.services.chat_service import ChatService
from app.application.services.chunking_service import ChunkingService
from app.application.services.grounding_service import GroundingService
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.prompt_builder import build_extraction_prompt  # noqa: F401 — re-exported
from app.application.services.retrieval_service import RetrievalService
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.domain.repositories.chat_repository import ChatRepository
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.infrastructure.pdf.document_reader import DocumentReader
from app.infrastructure.repositories.sqlite_chat_repository import SQLiteChatRepository
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore

# ── Module-level singletons ────────────────────────────────────────────────────
# Populated by initialise() during FastAPI lifespan startup.

_settings = None
_database: SQLiteDatabase | None = None
_document_repository: SQLiteDocumentRepository | None = None
_chat_repository: SQLiteChatRepository | None = None
_document_reader: DocumentReader | None = None
_embedding_client: OllamaEmbeddingClient | None = None
_llm_client: OllamaLLMClient | None = None
_vector_store: ChromaVectorStore | None = None
_chunking_service: ChunkingService | None = None
_text_cleaner: TextCleaner | None = None
_metadata_validator: MetadataValidator | None = None
_retrieval_service: RetrievalService | None = None
_grounding_service: GroundingService | None = None
_chat_service: ChatService | None = None
_extract_use_case: ExtractMetadataUseCase | None = None
_chat_use_case: ChatWithDocumentUseCase | None = None
_async_extraction_service: AsyncExtractionService | None = None


from app.application.services.intent_router import IntentRouter

# (imports...)

_intent_router: IntentRouter | None = None


def initialise() -> None:
    """Build the full dependency graph. Called once at application startup."""
    global _settings, _database, _document_repository, _chat_repository, _document_reader
    global _embedding_client, _llm_client, _vector_store
    global _chunking_service, _text_cleaner, _metadata_validator
    global _retrieval_service, _grounding_service, _chat_service, _intent_router
    global _extract_use_case, _chat_use_case, _async_extraction_service

    _settings = get_settings()

    # Infrastructure
    _database = SQLiteDatabase(_settings)
    _database.initialise()

    _document_repository = SQLiteDocumentRepository(_database)
    _chat_repository = SQLiteChatRepository(_database)
    _document_reader = DocumentReader()
    
    _embedding_client = OllamaEmbeddingClient(_settings)
    _llm_client = OllamaLLMClient(_settings)
    _vector_store = ChromaVectorStore(_settings, _embedding_client)

    # Application services
    _text_cleaner = TextCleaner()
    _chunking_service = ChunkingService(_settings)
    _metadata_validator = MetadataValidator()
    
    _retrieval_service = RetrievalService(_vector_store, _settings)
    _grounding_service = GroundingService(_settings)
    _chat_service = ChatService(_settings)
    _intent_router = IntentRouter()

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

    _chat_use_case = ChatWithDocumentUseCase(
        retrieval_service=_retrieval_service,
        grounding_service=_grounding_service,
        chat_service=_chat_service,
        intent_router=_intent_router,
        document_repository=_document_repository,
        chat_repository=_chat_repository,
        settings=_settings,
    )
    
    _async_extraction_service = AsyncExtractionService(
        use_case=_extract_use_case,
        settings=_settings,
        max_concurrent=2,
    )


# ── FastAPI Depends provider functions ────────────────────────────────────────

def get_database() -> SQLiteDatabase:
    assert _database is not None, "Container not initialised"
    return _database


def get_extract_use_case() -> ExtractMetadataUseCase:
    assert _extract_use_case is not None, "Container not initialised"
    return _extract_use_case


def get_chat_use_case() -> ChatWithDocumentUseCase:
    assert _chat_use_case is not None, "Container not initialised"
    return _chat_use_case


def get_document_repository() -> DocumentRepository:
    assert _document_repository is not None, "Container not initialised"
    return _document_repository


def get_chat_repository() -> ChatRepository:
    assert _chat_repository is not None, "Container not initialised"
    return _chat_repository


def get_async_extraction_service() -> AsyncExtractionService:
    assert _async_extraction_service is not None, "Container not initialised"
    return _async_extraction_service
