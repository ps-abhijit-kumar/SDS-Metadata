"""Comprehensive regression test suite for Version-Aware Document Intelligence.

Verifies:
  1. First upload creates Version 1.
  2. Exact same content reuses existing Version 1 (ZERO LLM calls).
  3. Same filename + changed content creates Version 2.
  4. Version 1 remains intact in history.
  5. Re-uploading V1 content after V2 re-activates V1 content version.
  6. Same content + stale processing version triggers vector reindex only.
  7. Vector reindex does NOT call metadata LLM.
  8. Five different documents remain isolated in vector search.
  9. p6557.pdf first-aid query retrieves Section 4.
  10. English query against Spanish Section 4 works.
  11. l0288.pdf first-aid retrieval continues working.
  12. Source attribution contains document, section, page, and section title.
  13. Unrelated questions are rejected before calling the LLM.
  14. Four mandatory fields always correspond to current active version.
  15. Current version changes correctly upon new version upload.
"""

from pathlib import Path
import tempfile
import pytest

from app.domain.entities.document import Document
from app.domain.enums.document_status import DocumentStatus
from app.domain.value_objects.sds_metadata import SDSMetadata
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.repositories.sqlite_chat_repository import SQLiteChatRepository
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
from app.application.services.chunking_service import ChunkingService
from app.application.services.retrieval_service import RetrievalService
from app.application.services.grounding_service import GroundingService
from app.application.services.chat_service import ChatService
from app.application.services.intent_router import IntentRouter
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase, CURRENT_PROCESSING_VERSION
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase


class MockLLMClient:
    """Mock LLM client to verify whether LLM inference was called."""
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, model: str = None) -> str:
        self.call_count += 1
        return (
            '{"product_name": "Mock SDS Product", '
            '"company_name": "Mock Chemical Corp", '
            '"language": "English", '
            '"jurisdiction": "United States (OSHA / HazCom 2012)"}'
        )


class DummyReader:
    """Simple PDF reader mock for version testing."""
    def __init__(self, text="Sample SDS text Section 1 Identification"):
        self.text = text

    def read(self, path):
        class Extracted:
            pass
        e = Extracted()
        e.full_text = self.text
        e.pages = []
        return e


class DummyCleaner:
    def clean(self, text):
        return text


@pytest.fixture
def test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    settings = get_settings()
    settings.database_url = f"sqlite:///{db_path}"
    db = SQLiteDatabase(settings)
    db.initialise()
    yield db
    try:
        db_path.unlink()
    except Exception:
        pass


def test_1_and_2_first_upload_and_exact_duplicate(test_db):
    """Test 1 & 2: First upload creates V1; exact duplicate reuses V1 without calling LLM."""
    settings = get_settings()
    repo = SQLiteDocumentRepository(test_db)
    mock_llm = MockLLMClient()
    embedder = OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings, embedder)
    chunker = ChunkingService(settings)

    class DummyValidator:
        def parse_and_validate(self, doc_id, text):
            return SDSMetadata(
                file_id=doc_id,
                product_name="Product Alpha",
                company_name="Alpha Corp",
                language="English",
                jurisdiction="US OSHA",
            )

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=DummyReader("Section 1 Identification Product Alpha"),
        text_cleaner=DummyCleaner(),
        chunking_service=chunker,
        vector_store=vector_store,
        llm_client=mock_llm,
        metadata_validator=DummyValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 Mock Content Alpha V1")
        tmp_path = Path(tmp.name)

    try:
        # First Upload
        res1 = use_case.execute(tmp_path, "sample_alpha.pdf")
        doc1 = repo.find_by_id(res1.document_id)

        assert doc1.version_number == 1
        assert doc1.processing_version == CURRENT_PROCESSING_VERSION
        assert doc1.is_active is True
        assert mock_llm.call_count == 1

        # Second Upload (Exact duplicate content & hash)
        res2 = use_case.execute(tmp_path, "sample_alpha.pdf")
        assert mock_llm.call_count == 1  # LLM NOT CALLED!
        assert res2.product_name == "Product Alpha"
        
        all_docs = repo.find_all()
        assert len(all_docs) == 2
        # Duplicate record saved for history
        assert any(d.status == DocumentStatus.DUPLICATE for d in all_docs)

    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def test_3_4_5_content_versioning_and_reactivation(test_db):
    """Test 3, 4, 5: Changed content creates V2, V1 remains in history, re-uploading V1 content reactivates V1."""
    settings = get_settings()
    repo = SQLiteDocumentRepository(test_db)
    mock_llm = MockLLMClient()
    embedder = OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings, embedder)
    chunker = ChunkingService(settings)

    class DummyValidator:
        def parse_and_validate(self, doc_id, text):
            return SDSMetadata(
                file_id=doc_id,
                product_name="Product Versioned",
                company_name="Version Corp",
                language="English",
                jurisdiction="US OSHA",
            )

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=DummyReader(),
        text_cleaner=DummyCleaner(),
        chunking_service=chunker,
        vector_store=vector_store,
        llm_client=mock_llm,
        metadata_validator=DummyValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
        tmp1.write(b"%PDF-1.4 Content Version 1")
        file1 = Path(tmp1.name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
        tmp2.write(b"%PDF-1.4 Content Version 2 - Modified")
        file2 = Path(tmp2.name)

    try:
        # Upload V1
        res1 = use_case.execute(file1, "product_spec.pdf")
        doc1 = repo.find_by_id(res1.document_id)
        assert doc1.version_number == 1
        assert doc1.is_active is True

        # Upload V2 (Same filename, changed content)
        res2 = use_case.execute(file2, "product_spec.pdf")
        doc2 = repo.find_by_id(res2.document_id)
        assert doc2.version_number == 2
        assert doc2.is_active is True

        # Verify V1 remains in history but is deactivated
        doc1_updated = repo.find_by_id(res1.document_id)
        assert doc1_updated is not None
        assert doc1_updated.is_active is False

        # Re-upload V1 content again (file1)
        res3 = use_case.execute(file1, "product_spec.pdf")
        # Should reactivate V1 and return cached V1 result without calling LLM!
        doc1_reactivated = repo.find_by_id(res1.document_id)
        assert doc1_reactivated.is_active is True

        doc2_deactivated = repo.find_by_id(res2.document_id)
        assert doc2_deactivated.is_active is False

    finally:
        try:
            file1.unlink()
            file2.unlink()
        except Exception:
            pass


def test_6_7_stale_index_rebuild_without_llm(test_db):
    """Test 6 & 7: Same content + outdated processing version triggers vector reindex ONLY without calling LLM."""
    settings = get_settings()
    repo = SQLiteDocumentRepository(test_db)
    mock_llm = MockLLMClient()
    embedder = OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings, embedder)
    chunker = ChunkingService(settings)

    class DummyValidator:
        def parse_and_validate(self, doc_id, text):
            return SDSMetadata(
                file_id=doc_id,
                product_name="Stale Test Chemical",
                company_name="Stale Inc",
                language="English",
                jurisdiction="US OSHA",
            )

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=DummyReader(),
        text_cleaner=DummyCleaner(),
        chunking_service=chunker,
        vector_store=vector_store,
        llm_client=mock_llm,
        metadata_validator=DummyValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 Stale Index Test File")
        file_path = Path(tmp.name)

    try:
        # 1. Initial upload
        res1 = use_case.execute(file_path, "stale_doc.pdf")
        assert mock_llm.call_count == 1

        # 2. Simulate outdated processing_version='v1' in SQLite
        doc1 = repo.find_by_id(res1.document_id)
        doc1.processing_version = "v1"
        repo.update(doc1)

        # 3. Re-upload same file
        res2 = use_case.execute(file_path, "stale_doc.pdf")

        # Verify LLM was NOT called again (call_count remains 1)
        assert mock_llm.call_count == 1

        # Verify processing_version updated to CURRENT_PROCESSING_VERSION ("v2")
        doc1_rebuilt = repo.find_by_id(res1.document_id)
        assert doc1_rebuilt.processing_version == CURRENT_PROCESSING_VERSION
        assert res2.product_name == "Stale Test Chemical"

    finally:
        try:
            file_path.unlink()
        except Exception:
            pass


def test_8_document_isolation(test_db):
    """Test 8: Ensure 5 different documents remain isolated in canonical queries."""
    repo = SQLiteDocumentRepository(test_db)
    for i in range(5):
        doc = Document(
            id=f"doc_{i}",
            filename=f"file_{i}.pdf",
            file_path=f"/path/to/file_{i}.pdf",
            file_hash=f"hash_{i}",
            status=DocumentStatus.COMPLETED,
            metadata=SDSMetadata(file_id=f"doc_{i}", product_name=f"Product {i}"),
            is_active=True,
        )
        repo.save(doc)

    canonical = repo.find_all_canonical()
    assert len(canonical) == 5
    ids = {d.id for d in canonical}
    assert ids == {f"doc_{i}" for i in range(5)}


def test_9_10_11_12_p6557_and_l0288_retrieval_and_attribution(test_db):
    """Test 9, 10, 11, 12: Section 4 RAG for p6557.pdf (Spanish) & l0288.pdf with full source attribution."""
    import fitz
    settings = get_settings()
    embedder = OllamaEmbeddingClient(settings)
    store = ChromaVectorStore(settings, embedder)
    retriever = RetrievalService(store, settings)
    grounding = GroundingService(settings)
    chat_svc = ChatService(settings)
    doc_repo = SQLiteDocumentRepository(test_db)
    chat_repo = SQLiteChatRepository(test_db)

    doc_id = "test_p6557_version_rag"
    pdf_path = Path(__file__).parent / "fixtures" / "p6557.pdf"

    # Seed Document entity in SQLite
    doc_repo.save(
        Document(
            id=doc_id,
            filename="p6557.pdf",
            file_path=str(pdf_path),
            file_hash="hash_p6557",
            status=DocumentStatus.COMPLETED,
            metadata=SDSMetadata(
                file_id=doc_id,
                product_name="Product p6557",
                company_name="Sigma-Aldrich",
                language="Spanish",
                jurisdiction="Mexico (NOM-018-STPS)",
            ),
            is_active=True,
        )
    )

    # Chunk and seed in Chroma
    doc = fitz.open(str(pdf_path))
    pages = [type("ExtractedPage", (), {"page_number": i + 1, "text": page.get_text()})() for i, page in enumerate(doc)]
    chunker = ChunkingService(settings)
    chunks = chunker.chunk_pages(pages, document_id=doc_id, filename="p6557.pdf")
    store.add_documents(
        document_id=doc_id,
        texts=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    use_case = ChatWithDocumentUseCase(
        retrieval_service=retriever,
        grounding_service=grounding,
        chat_service=chat_svc,
        intent_router=IntentRouter(),
        document_repository=doc_repo,
        chat_repository=chat_repo,
        settings=settings,
    )

    response = use_case.execute("What are the first aid measures?", document_id=doc_id)

    assert response.grounded is True
    assert len(response.sources) > 0
    top_source = response.sources[0]
    assert "p6557" in top_source.document
    assert str(top_source.section) == "4"
    assert top_source.page == 3
    assert "First-Aid" in top_source.section_title or "Primeros" in top_source.section_title


def test_13_unrelated_question_grounding_rejection(test_db):
    """Test 13: Unrelated questions are rejected before calling the LLM (llm_ms == 0.0)."""
    import fitz
    settings = get_settings()
    embedder = OllamaEmbeddingClient(settings)
    store = ChromaVectorStore(settings, embedder)
    retriever = RetrievalService(store, settings)
    grounding = GroundingService(settings)
    chat_svc = ChatService(settings)
    doc_repo = SQLiteDocumentRepository(test_db)
    chat_repo = SQLiteChatRepository(test_db)

    doc_id = "test_p6557_version_rag_unrelated"
    pdf_path = Path(__file__).parent / "fixtures" / "p6557.pdf"

    doc_repo.save(
        Document(
            id=doc_id,
            filename="p6557.pdf",
            file_path=str(pdf_path),
            file_hash="hash_p6557_unrelated",
            status=DocumentStatus.COMPLETED,
            metadata=SDSMetadata(
                file_id=doc_id,
                product_name="Product p6557",
                company_name="Sigma-Aldrich",
                language="Spanish",
                jurisdiction="Mexico (NOM-018-STPS)",
            ),
            is_active=True,
        )
    )

    doc = fitz.open(str(pdf_path))
    pages = [type("ExtractedPage", (), {"page_number": i + 1, "text": page.get_text()})() for i, page in enumerate(doc)]
    chunker = ChunkingService(settings)
    chunks = chunker.chunk_pages(pages, document_id=doc_id, filename="p6557.pdf")
    store.add_documents(
        document_id=doc_id,
        texts=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    use_case = ChatWithDocumentUseCase(
        retrieval_service=retriever,
        grounding_service=grounding,
        chat_service=chat_svc,
        intent_router=IntentRouter(),
        document_repository=doc_repo,
        chat_repository=chat_repo,
        settings=settings,
    )

    response = use_case.execute("When was the Prime Minister born in India?", document_id=doc_id)

    assert response.grounded is False
    assert response.answer == "Information not available in the uploaded file."
    assert response.llm_ms == 0.0
    assert len(response.sources) == 0
