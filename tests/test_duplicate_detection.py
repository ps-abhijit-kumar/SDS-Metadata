"""Automated tests for SHA-256 duplicate detection and versioning behavior."""

import hashlib
import tempfile
from pathlib import Path
import pytest

from app.application.services.chunking_service import ChunkingService
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.prompt_builder import build_extraction_prompt
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.domain.entities.document import Document
from app.domain.enums.document_status import DocumentStatus
from app.domain.value_objects.sds_metadata import SDSMetadata
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository


class MockReader:
    def read(self, file_path: Path):
        class Extracted:
            full_text = "Section 1: Product Acetone. Section 15: OSHA."
            pages = []
            page_count = 1
        return Extracted()


class MockVectorStore:
    def add_documents(self, document_id, texts, metadatas=None):
        pass

    def similarity_search(self, queries, document_id, k=2):
        return ["Section 1: Product Acetone.", "Section 15: OSHA."]


class MockLLM:
    def generate(self, prompt, **kwargs):
        return "language: English\njurisdiction: United States (OSHA / HazCom 2012)\ncompany name: Test Chem Co\nproduct name: Acetone"


@pytest.fixture
def temp_db(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/test.db")
    db = SQLiteDatabase(settings)
    db.initialise()
    return db, settings


def test_sha256_duplicate_detection(temp_db):
    db, settings = temp_db
    repo = SQLiteDocumentRepository(db)
    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=MockReader(),
        text_cleaner=TextCleaner(),
        chunking_service=ChunkingService(settings),
        vector_store=MockVectorStore(),
        llm_client=MockLLM(),
        metadata_validator=MetadataValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 Mock PDF Content For Duplicate Check")
        tmp_path = Path(tmp.name)

    try:
        # First upload
        res1 = use_case.execute(tmp_path, "acetone.pdf")
        assert res1.status == DocumentStatus.COMPLETED
        assert res1.product_name == "Acetone"

        # Second upload with identical bytes
        res2 = use_case.execute(tmp_path, "acetone.pdf")
        assert res2.status == DocumentStatus.DUPLICATE
        assert res2.product_name == "Acetone"

    finally:
        tmp_path.unlink(missing_ok=True)


def test_same_filename_different_content(temp_db):
    db, settings = temp_db
    repo = SQLiteDocumentRepository(db)
    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=MockReader(),
        text_cleaner=TextCleaner(),
        chunking_service=ChunkingService(settings),
        vector_store=MockVectorStore(),
        llm_client=MockLLM(),
        metadata_validator=MetadataValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1:
        tmp1.write(b"%PDF-1.4 Content Version A")
        path1 = Path(tmp1.name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
        tmp2.write(b"%PDF-1.4 Content Version B")
        path2 = Path(tmp2.name)

    try:
        res1 = use_case.execute(path1, "acetone.pdf")
        assert res1.status == DocumentStatus.COMPLETED

        res2 = use_case.execute(path2, "acetone.pdf")
        assert res2.status == DocumentStatus.COMPLETED
        assert res1.document_id != res2.document_id

    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)


def test_10_duplicate_returns_canonical_language_name(temp_db):
    class SpanishLLM:
        def generate(self, prompt, **kwargs):
            return "language: es\njurisdiction: European Union (REACH / CLP)\ncompany name: Sigma-Aldrich\nproduct name: Lipid Mixture 1"

    db, settings = temp_db
    repo = SQLiteDocumentRepository(db)
    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=MockReader(),
        text_cleaner=TextCleaner(),
        chunking_service=ChunkingService(settings),
        vector_store=MockVectorStore(),
        llm_client=SpanishLLM(),
        metadata_validator=MetadataValidator(),
        settings=settings,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 Content For Spanish Duplicate Check")
        tmp_path = Path(tmp.name)

    try:
        res1 = use_case.execute(tmp_path, "l0288.pdf")
        assert res1.status == DocumentStatus.COMPLETED
        assert res1.language == "Spanish"  # NOT "es"!

        # Duplicate upload
        res2 = use_case.execute(tmp_path, "l0288.pdf")
        assert res2.status == DocumentStatus.DUPLICATE
        assert res2.language == "Spanish"  # MUST BE CANONICAL "Spanish", NOT "es"!

    finally:
        tmp_path.unlink(missing_ok=True)
