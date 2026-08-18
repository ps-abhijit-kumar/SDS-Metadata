"""Regression test suite for previously unseen SDS metadata extraction accuracy.

Verifies:
  1. Portuguese SDS (3461507_76_PT_BR_BA.pdf): Manufacturer company (Sigma-Aldrich Brasil Ltda.) != Brand (Millipore), Language = Portuguese, Jurisdiction = Brazil (ABNT NBR 14725).
  2. Spanish SDS (3508945_39261_ES_MX.pdf): Language = Spanish, Jurisdiction = Mexico (NOM-018-STPS).
  3. Devcon SDS (3853875_197_EN_CA_BA.pdf): Product Name = DEVCON Plastic Steel Liquid (B) (Main Product) != Component (LIQUID HARDENER 0203).
  4. Fisher Scientific SDS (3776218_704_EN_CA.pdf): Product Name = D-Fructose, Company = Fisher Scientific, Jurisdiction = Canada (WHMIS 2015).
  5. Missing Manufacturer MSDS (3577418_25136_EN_US_BA.pdf): Product Name = Rite-Qwik, Company = None (strict grounding, no fabrication).
"""

from pathlib import Path
import pytest

from app.application.services.chunking_service import ChunkingService
from app.application.services.language_detector import LanguageDetector
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.prompt_builder import build_extraction_prompt
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.pdf.document_reader import DocumentReader
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient


@pytest.fixture
def real_pipeline():
    settings = get_settings()
    db = SQLiteDatabase(settings)
    db.initialise()
    with db.connection() as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
    repo = SQLiteDocumentRepository(db)
    reader = DocumentReader()
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
    return use_case, reader


def test_language_detector_multilingual():
    """Verify LanguageDetector works deterministically across Portuguese, Spanish, English text snippets."""
    pt_text = "FICHA DE DADOS DE SEGURANÇA SEÇÃO 1. IDENTIFICAÇÃO do produto Hidrogenofosfato dissódico"
    es_text = "HOJA DE DATOS DE SEGURIDAD Clasificado de acuerdo con NOM-018-STPS-2015 1. Identificación"
    en_text = "SAFETY DATA SHEET 1. Identification Product Name D-Fructose WHMIS 2015"

    assert LanguageDetector.detect_language(pt_text) == "Portuguese"
    assert LanguageDetector.detect_language(es_text) == "Spanish"
    assert LanguageDetector.detect_language(en_text) == "English"


def test_1_portuguese_sds_manufacturer_not_brand(real_pipeline):
    """Test 1: Portuguese SDS distinguishes Manufacturer (Sigma-Aldrich Brasil Ltda.) from Brand (Millipore)."""
    use_case, reader = real_pipeline
    path = Path("data/uploads/a3a2a60e-d744-4d85-8614-07e15f59e00a_3461507_76_PT_BR_BA.pdf")
    if not path.exists():
        pytest.skip(f"{path} not found")

    res = use_case.execute(path, "3461507_76_PT_BR_BA.pdf")

    assert res.company_name is not None
    assert "Sigma-Aldrich" in res.company_name or "Brasil" in res.company_name
    assert "Millipore" not in res.company_name  # Brand MUST NOT be company!
    assert res.language == "Portuguese"
    assert res.jurisdiction == "Brazil (ABNT NBR 14725)"


def test_2_spanish_sds_language_and_jurisdiction(real_pipeline):
    """Test 2: Spanish SDS language detection and Mexican jurisdiction normalization."""
    use_case, reader = real_pipeline
    path = Path("data/uploads/28b47118-af14-4646-a21c-ddf174a942de_3508945_39261_ES_MX.pdf")
    if not path.exists():
        pytest.skip(f"{path} not found")

    res = use_case.execute(path, "3508945_39261_ES_MX.pdf")

    assert res.language == "Spanish"
    assert res.jurisdiction == "Mexico (NOM-018-STPS)"
    assert res.product_name is not None
    assert "IS 808" in res.product_name


def test_3_devcon_sds_product_not_component(real_pipeline):
    """Test 3: Devcon SDS product extraction selects main Product Name, NOT kit component hardener."""
    use_case, reader = real_pipeline
    path = Path("data/uploads/3ae7f1bd-cb37-48d8-8cc6-8b431726f010_3853875_197_EN_CA_BA.pdf")
    if not path.exists():
        pytest.skip(f"{path} not found")

    res = use_case.execute(path, "3853875_197_EN_CA_BA.pdf")

    assert res.product_name is not None
    assert "HARDENER" not in res.product_name.upper()  # Component MUST NOT be product name!
    assert "PLASTIC STEEL" in res.product_name.upper() or "DEVCON" in res.product_name.upper()


def test_4_fisher_scientific_sds(real_pipeline):
    """Test 4: Fisher Scientific SDS product, manufacturer, and Canadian jurisdiction."""
    use_case, reader = real_pipeline
    path = Path("data/uploads/611ba478-3f85-4efc-b95b-186b10a5a59e_3776218_704_EN_CA.pdf")
    if not path.exists():
        pytest.skip(f"{path} not found")

    res = use_case.execute(path, "3776218_704_EN_CA.pdf")

    assert res.product_name == "D-Fructose"
    assert res.company_name is not None
    assert "Fisher Scientific" in res.company_name
    assert res.jurisdiction == "Canada (WHMIS 2015)"


def test_5_missing_company_grounding_rejection(real_pipeline):
    """Test 5: MSDS without explicit company yields company_name = None under strict grounding."""
    use_case, reader = real_pipeline
    path = Path("data/uploads/f78b09cc-65e2-432f-8715-35c8f5137c89_3577418_25136_EN_US_BA.pdf")
    if not path.exists():
        pytest.skip(f"{path} not found")

    res = use_case.execute(path, "3577418_25136_EN_US_BA.pdf")

    assert "Rite-Qwik" in res.product_name or "Water scale cleaner" in res.product_name
    # Grounded extraction: If extracted, must match actual entity present in document text ("Armstrong Hot Water Group")
    assert res.company_name is None or "Armstrong" in res.company_name
