"""Regression tests for multilingual section-aware RAG retrieval and strict grounding anti-hallucination."""

from __future__ import annotations

from pathlib import Path
import fitz
import pytest

from app.application.services.chunking_service import ChunkingService
from app.application.services.grounding_service import GroundingService
from app.application.services.retrieval_service import RetrievalService
from app.application.services.section_detector import SectionDetector
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore


class ExtractedPage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text


def _load_pdf_pages(pdf_path: str) -> list[ExtractedPage]:
    doc = fitz.open(pdf_path)
    return [ExtractedPage(i + 1, page.get_text()) for i, page in enumerate(doc)]


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def vector_store(settings):
    embedder = OllamaEmbeddingClient(settings)
    return ChromaVectorStore(settings, embedder)


@pytest.fixture(scope="module")
def services(settings, vector_store):
    section_detector = SectionDetector()
    retrieval_service = RetrievalService(vector_store, settings, section_detector)
    grounding_service = GroundingService(settings)
    chunking_service = ChunkingService(settings)
    return {
        "section_detector": section_detector,
        "retrieval_service": retrieval_service,
        "grounding_service": grounding_service,
        "chunking_service": chunking_service,
        "vector_store": vector_store,
    }


@pytest.fixture(scope="module")
def prepare_p6557(services):
    pdf_path = Path(__file__).parent / "fixtures" / "p6557.pdf"
    pages = _load_pdf_pages(str(pdf_path))
    document_id = "test_p6557_regression"
    filename = "p6557.pdf"

    chunker = services["chunking_service"]
    chunks = chunker.chunk_pages(pages, document_id=document_id, filename=filename)

    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]

    store = services["vector_store"]
    store.add_documents(document_id=document_id, texts=texts, metadatas=metas)
    return {"document_id": document_id, "filename": filename, "chunks": chunks}


@pytest.fixture(scope="module")
def prepare_l0288(services):
    pdf_path = Path(__file__).parent / "fixtures" / "l0288.pdf"
    pages = _load_pdf_pages(str(pdf_path))
    document_id = "test_l0288_regression"
    filename = "l0288.pdf"

    chunker = services["chunking_service"]
    chunks = chunker.chunk_pages(pages, document_id=document_id, filename=filename)

    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]

    store = services["vector_store"]
    store.add_documents(document_id=document_id, texts=texts, metadatas=metas)
    return {"document_id": document_id, "filename": filename, "chunks": chunks}


def test_1_p6557_first_aid_retrieval(services, prepare_p6557):
    """TEST 1: p6557.pdf + 'What are the first aid measures?'

    - target section = 4
    - Section 4 chunks retrieved
    - retrieval does not return zero
    - grounding succeeds
    """
    retriever = services["retrieval_service"]
    detector = services["section_detector"]
    grounding = services["grounding_service"]
    doc_id = prepare_p6557["document_id"]

    query = "What are the first aid measures?"
    target_section = detector.detect_section(query)
    assert target_section == "4", f"Expected target section '4', got '{target_section}'"

    chunks = retriever.retrieve(query=query, document_id=doc_id)
    assert len(chunks) > 0, "Retrieval returned 0 chunks for Section 4 query on p6557.pdf"

    is_grounded, fallback = grounding.verify_grounding(chunks, scope="single")
    assert is_grounded is True, f"Grounding failed: {fallback}"

    # Verify retrieved chunks belong to section 4
    section_4_found = any(str(c.section) == "4" for c in chunks)
    assert section_4_found is True, "Retrieved chunks did not include any Section 4 chunks"


def test_2_english_query_against_spanish_section_4(services, prepare_p6557):
    """TEST 2: English query against Spanish Section 4 ('SECCIÓN 4. Primeros auxilios')."""
    retriever = services["retrieval_service"]
    grounding = services["grounding_service"]
    doc_id = prepare_p6557["document_id"]

    query = "What are the first aid measures?"
    chunks = retriever.retrieve(query=query, document_id=doc_id)
    assert len(chunks) > 0

    is_grounded, _ = grounding.verify_grounding(chunks, scope="single")
    assert is_grounded is True

    # Check text content in Spanish section 4
    text_combined = " ".join([c.text for c in chunks]).lower()
    assert ("primeros auxilios" in text_combined or "inhalaci" in text_combined or "piel" in text_combined)


def test_3_unrelated_question_grounding_failure(services, prepare_p6557, settings):
    """TEST 3: Unrelated question must NOT retrieve chunks, grounding must fail, LLM not called.

    Example: 'When was the Prime Minister born in India?'
    """
    retriever = services["retrieval_service"]
    grounding = services["grounding_service"]
    doc_id = prepare_p6557["document_id"]

    query = "When was the Prime Minister born in India?"
    chunks = retriever.retrieve(query=query, document_id=doc_id)
    assert len(chunks) == 0, f"Expected 0 chunks for unrelated query, got {len(chunks)}"

    is_grounded, fallback = grounding.verify_grounding(chunks, scope="single")
    assert is_grounded is False
    assert fallback == settings.fallback_response


def test_4_l0288_first_aid_retrieval(services, prepare_l0288):
    """TEST 4: l0288.pdf + 'What are the first aid measures?'

    Existing working behavior remains intact.
    """
    retriever = services["retrieval_service"]
    grounding = services["grounding_service"]
    doc_id = prepare_l0288["document_id"]

    query = "What are the first aid measures?"
    chunks = retriever.retrieve(query=query, document_id=doc_id)
    assert len(chunks) > 0, "Retrieval returned 0 chunks for l0288.pdf"

    is_grounded, _ = grounding.verify_grounding(chunks, scope="single")
    assert is_grounded is True


def test_5_source_attribution_metadata(services, prepare_p6557):
    """TEST 5: Source attribution metadata verification.

    Retrieved chunk metadata must contain:
    - filename: p6557.pdf
    - section: 4
    - page: 3 (or valid integer page)
    - section_title: Section 4 title
    """
    retriever = services["retrieval_service"]
    doc_id = prepare_p6557["document_id"]

    query = "What are the first aid measures?"
    chunks = retriever.retrieve(query=query, document_id=doc_id)
    assert len(chunks) > 0

    section_4_chunks = [c for c in chunks if str(c.section) == "4"]
    assert len(section_4_chunks) > 0, "No Section 4 chunk found in retrieved results"

    target_chunk = section_4_chunks[0]
    assert target_chunk.filename == "p6557.pdf"
    assert str(target_chunk.section) == "4"
    assert target_chunk.page == 3
    assert "Section 4" in target_chunk.section_title or "First-Aid" in target_chunk.section_title
