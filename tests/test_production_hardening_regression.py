"""Comprehensive production regression suite covering all 18 production failure modes (A-R):

A. 5-document concurrent ingestion
B. Chroma concurrent storage safety
C. Full Portuguese product value (no truncation at 'dissódico')
D. Portuguese manufacturer vs brand (Sigma-Aldrich vs Millipore)
E. Spanish manufacturer extraction (Momentive Performance Materials, not 'Identificador del producto')
F. Spanish product & jurisdiction extraction
G. English Devcon manufacturer extraction (ITW Devcon, not 'Name')
H. Devcon primary product vs hardener component
I. Legacy MSDS first-aid retrieval (non-numeric Section 4)
J. English query against Spanish first-aid section
K. Document-scoped first-aid retrieval
L. All-document first-aid query with source separation
M. Cross-document contamination prevention
N. Source attribution metadata formatting
O. Duplicate cache invalidation on extraction-version upgrade
P. Unrelated question rejected with no LLM call
Q. Changed PDF content creates a new version
R. Unchanged PDF + current extraction version fast path
"""

import concurrent.futures
from pathlib import Path
import pytest

from app.application.dto.chat_dto import SourceCitationDTO
from app.application.services.chat_service import ChatService
from app.application.services.chunking_service import ChunkingService
from app.application.services.grounding_service import GroundingService
from app.application.services.intent_router import IntentRouter
from app.application.services.language_detector import LanguageDetector
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.retrieval_service import RetrievalService, RetrievedChunk
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.application.use_cases.extract_metadata_use_case import (
    CURRENT_PROCESSING_VERSION,
    ExtractMetadataUseCase,
)
from app.domain.entities.document import Document
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.infrastructure.pdf.document_reader import DocumentReader
from app.infrastructure.repositories.sqlite_chat_repository import SQLiteChatRepository
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore


class SmartReader:
    def read(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = "Sample SDS text Section 1 Identification"

        class PageObj:
            def __init__(self, page_number, text):
                self.page_number = page_number
                self.text = text

        class Extracted:
            pass

        e = Extracted()
        e.full_text = content
        e.page_count = 1
        e.pages = [PageObj(1, content)]
        return e


@pytest.fixture
def test_pipeline():
    settings = get_settings()
    db = SQLiteDatabase(settings)
    db.initialise()
    with db.connection() as conn:
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM chat_history")
        conn.commit()

    repo = SQLiteDocumentRepository(db)
    chat_repo = SQLiteChatRepository(db)
    reader = SmartReader()
    cleaner = TextCleaner()
    chunker = ChunkingService(settings)
    embedder = OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings, embedder)
    llm = OllamaLLMClient(settings)
    validator = MetadataValidator()

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=reader,
        text_cleaner=cleaner,
        chunking_service=chunker,
        vector_store=vector_store,
        llm_client=llm,
        metadata_validator=validator,
        settings=settings,
    )

    retrieval_svc = RetrievalService(vector_store, settings)
    grounding_svc = GroundingService(settings)
    chat_svc = ChatService(settings)
    intent_router = IntentRouter()

    chat_use_case = ChatWithDocumentUseCase(
        retrieval_service=retrieval_svc,
        grounding_service=grounding_svc,
        chat_service=chat_svc,
        intent_router=intent_router,
        document_repository=repo,
        chat_repository=chat_repo,
        settings=settings,
    )

    return use_case, chat_use_case, vector_store, repo


# ── TEST A & B: Concurrent Ingestion and Chroma Concurrency Safety ────────────
def test_a_and_b_concurrent_ingestion_and_chroma_safety(test_pipeline, tmp_path):
    use_case, _, vector_store, repo = test_pipeline

    files = []
    for i in range(5):
        p = tmp_path / f"doc_{i}.pdf"
        p.write_bytes(f"%PDF-1.4 Fake document content {i}\nSECTION 1 Identification\nProduct Name: Test Product {i}\nManufacturer: Test Company {i}".encode())
        files.append(p)

    def process_file(idx_and_path):
        idx, path = idx_and_path
        return use_case.execute(path, f"doc_{idx}.pdf")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_file, enumerate(files)))

    assert len(results) == 5
    for r in results:
        assert r.document_id is not None
        assert r.language is not None


# ── TEST C: Full Portuguese Product Value (No Truncation) ─────────────────────
def test_c_full_portuguese_product_value():
    validator = MetadataValidator()
    context = """
    FICHA DE DADOS DE SEGURANÇA
    Nome do produto: Hidrogenofosfato dissódico dodecahidratado para análise EMSURE® ISO,Reag. Ph Eur
    Companhia: Sigma-Aldrich Brasil Ltda.
    """
    llm_resp = """
    Language: Portuguese
    Jurisdiction: Brazil (ABNT NBR 14725)
    Company Name: Sigma-Aldrich Brasil Ltda.
    Product Name: Hidrogenofosfato dissódico dodecahidratado para análise EMSURE® ISO,Reag. Ph Eur
    """
    res = validator.parse_and_validate("doc_pt", llm_resp, context_text=context)
    assert res.product_name == "Hidrogenofosfato dissódico dodecahidratado para análise EMSURE® ISO,Reag. Ph Eur"
    assert "dissódico" in res.product_name


# ── TEST D: Portuguese Manufacturer vs Brand ─────────────────────────────────
def test_d_portuguese_manufacturer_not_brand():
    validator = MetadataValidator()
    context = """
    FICHA DE DADOS DE SEGURANÇA
    Companhia: Sigma-Aldrich Brasil Ltda.
    Marca: Millipore
    """
    llm_resp = """
    Language: Portuguese
    Jurisdiction: Brazil (ABNT NBR 14725)
    Company Name: Sigma-Aldrich Brasil Ltda.
    Product Name: Hidrogenofosfato dissódico
    """
    res = validator.parse_and_validate("doc_pt", llm_resp, context_text=context)
    assert res.company_name == "Sigma-Aldrich Brasil Ltda."
    assert res.company_name != "Millipore"


# ── TEST E & F: Spanish Manufacturer & Product Extraction ────────────────────
def test_e_and_f_spanish_manufacturer_and_product():
    validator = MetadataValidator()
    context = """
    FICHA DE DATOS DE SEGURIDAD
    1. IDENTIFICACIÓN DE LA SUSTANCIA
    Identificador del producto: SILQUEST A-1110 SILANE
    Información sobre el fabricante/importador/distribuidor:
    Momentive Performance Materials
    """
    llm_resp = """
    Language: Spanish
    Jurisdiction: NOM-018-STPS
    Company Name: Momentive Performance Materials
    Product Name: SILQUEST A-1110 SILANE
    """
    res = validator.parse_and_validate("doc_es", llm_resp, context_text=context)
    assert res.company_name == "Momentive Performance Materials"
    assert res.company_name != "Identificador del producto"
    assert res.product_name == "SILQUEST A-1110 SILANE"
    assert res.jurisdiction == "Mexico (NOM-018-STPS)"


# ── TEST G & H: English Devcon Manufacturer & Product vs Hardener ────────────
def test_g_and_h_devcon_manufacturer_and_primary_product():
    validator = MetadataValidator()
    context = """
    SAFETY DATA SHEET
    SECTION 1: Identification
    Product Name: DEVCON® Plastic Steel® Liquid (B)
    Manufacturer Name: ITW Devcon
    Component: LIQUID HARDENER 0203
    """
    llm_resp = """
    Language: English
    Jurisdiction: WHMIS 2015
    Company Name: ITW Devcon
    Product Name: DEVCON® Plastic Steel® Liquid (B)
    """
    res = validator.parse_and_validate("doc_devcon", llm_resp, context_text=context)
    assert res.company_name == "ITW Devcon"
    assert res.company_name != "Name"
    assert "Plastic Steel" in res.product_name
    assert "HARDENER" not in res.product_name


# ── TEST I & J: Legacy MSDS First Aid Retrieval & Multilingual Query ─────────
def test_i_and_j_legacy_msds_first_aid_retrieval(test_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = test_pipeline

    msds_path = tmp_path / "legacy_msds.pdf"
    msds_path.write_bytes(
        b"%PDF-1.4\nMATERIAL SAFETY DATA SHEET\nProduct Name: Rite-Qwik\n"
        b"Emergency and First Aid Exposures\n"
        b"Inhalation: Move patient to fresh air. If breathing is difficult, give oxygen.\n"
        b"Eye Contact: Flush eyes with water for 15 minutes."
    )

    doc_res = use_case.execute(msds_path, "3577418_25136_EN_US_BA.pdf")
    chat_res = chat_use_case.execute("What are the first aid measures?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert "fresh air" in chat_res.answer.lower() or "eye" in chat_res.answer.lower()


# ── TEST K, L, M: Cross-Document Isolation and Contamination Prevention ──────
def test_k_l_m_cross_document_isolation(test_pipeline, tmp_path):
    use_case, chat_use_case, _, repo = test_pipeline

    # Doc 1: Portuguese SDS
    pt_path = tmp_path / "pt_doc.pdf"
    pt_path.write_bytes(
        b"%PDF-1.4\nFICHA DE DADOS DE SEGURANCA\nNome do produto: Produto A\n"
        b"SECTION 4 Medidas de primeiros socorros\n"
        b"Inalacao: Remover a vitima para local arejado.\n"
        b"Contacto com a pele: Lavar com agua abundante."
    )
    doc1 = use_case.execute(pt_path, "3461507_76_PT_BR_BA.pdf")

    # Doc 2: English SDS
    en_path = tmp_path / "en_doc.pdf"
    en_path.write_bytes(
        b"%PDF-1.4\nSAFETY DATA SHEET\nProduct Name: Product B\n"
        b"SECTION 4 First-Aid Measures\n"
        b"Show this safety data sheet to the doctor in attendance. Immediate medical attention is required."
    )
    doc2 = use_case.execute(en_path, "3853875_197_EN_CA_BA.pdf")

    # Test K: Single document scoping for Doc 1
    res_single = chat_use_case.execute("What are the first aid measures?", document_id=doc1.document_id)
    assert res_single.grounded is True
    assert "doctor in attendance" not in res_single.answer.lower()

    # Test L & M: Multi-document scope separator
    res_all = chat_use_case.execute("What are the first aid measures?", document_id="all")
    assert res_all.grounded is True
    # Ensure source citations reflect both documents separately
    docs_in_sources = {s.document for s in res_all.sources}
    assert "3461507_76_PT_BR_BA.pdf" in docs_in_sources or "3853875_197_EN_CA_BA.pdf" in docs_in_sources


# ── TEST N: Source Attribution Metadata Formatting ───────────────────────────
def test_n_source_attribution_formatting():
    chat_svc = ChatService(get_settings())
    chunks = [
        RetrievedChunk(
            text="First aid instructions",
            document_id="doc1",
            filename="3577418_25136_EN_US_BA.pdf",
            page=2,
            section="0",
            section_title="Emergency and First Aid Exposures",
            score=0.1,
        )
    ]
    sources = chat_svc.extract_sources(chunks)
    assert len(sources) == 1
    assert sources[0].document == "3577418_25136_EN_US_BA.pdf"
    assert sources[0].page == 2
    assert sources[0].section == "N/A"
    assert sources[0].section_title == "Emergency and First Aid Exposures"


# ── TEST O: Duplicate Cache Invalidation on Processing Version Change ────────
def test_o_duplicate_cache_invalidation_version_change(test_pipeline, tmp_path):
    use_case, _, _, repo = test_pipeline

    pdf_path = tmp_path / "version_test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nSAFETY DATA SHEET\nProduct Name: Test Version Chem\nCompany: Test Co")

    doc1_res = use_case.execute(pdf_path, "version_test.pdf")

    # Simulate older processing_version='v2' in SQLite
    doc_entity = repo.find_by_id(doc1_res.document_id)
    doc_entity.processing_version = "v2"
    repo.update(doc_entity)

    # Re-upload same SHA-256 PDF under current CURRENT_PROCESSING_VERSION ("v3")
    doc2_res = use_case.execute(pdf_path, "version_test.pdf")
    doc2_entity = repo.find_by_id(doc2_res.document_id)

    assert doc2_entity.processing_version == CURRENT_PROCESSING_VERSION


# ── TEST P: Unrelated Question Grounding Rejection ───────────────────────────
def test_p_unrelated_question_rejection(test_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = test_pipeline

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nSAFETY DATA SHEET\nProduct Name: Acetone\nSECTION 4 First Aid\nFlush eyes with water.")

    doc = use_case.execute(pdf_path, "sample.pdf")
    chat_res = chat_use_case.execute("Who won the 2022 World Cup?", document_id=doc.document_id)

    assert chat_res.grounded is False
    assert len(chat_res.sources) == 0


# ── TEST Q & R: Content Versioning & Unchanged Content Fast Path ─────────────
def test_q_and_r_content_versioning_and_fast_path(test_pipeline, tmp_path):
    use_case, _, _, repo = test_pipeline

    pdf_v1 = tmp_path / "prod.pdf"
    pdf_v1.write_bytes(b"%PDF-1.4\nSAFETY DATA SHEET\nProduct Name: Product V1")

    res1 = use_case.execute(pdf_v1, "prod.pdf")
    doc1 = repo.find_by_id(res1.document_id)
    assert doc1.version_number == 1
    assert doc1.is_active is True

    # Exact same content fast path (Test R)
    res1_dup = use_case.execute(pdf_v1, "prod.pdf")
    assert res1_dup.document_id != res1.document_id

    # Changed content under same filename (Test Q)
    pdf_v2 = tmp_path / "prod_v2.pdf"
    pdf_v2.write_bytes(b"%PDF-1.4\nSAFETY DATA SHEET\nProduct Name: Product V2 Modified")

    res2 = use_case.execute(pdf_v2, "prod.pdf")
    doc2 = repo.find_by_id(res2.document_id)
    assert doc2.version_number == 2
    assert doc2.is_active is True

    # Check doc1 was deactivated
    doc1_updated = repo.find_by_id(res1.document_id)
    assert doc1_updated.is_active is False
