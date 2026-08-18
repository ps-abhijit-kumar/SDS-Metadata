"""End-to-end test verifying multi-document processing for 5+ PDFs."""

import tempfile
from pathlib import Path
import pytest

from app.application.services.chunking_service import ChunkingService
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.domain.enums.document_status import DocumentStatus
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository


class MultiDocReader:
    def read(self, file_path: Path):
        content = file_path.read_text()
        class Extracted:
            full_text = content
            pages = []
            page_count = 1
        return Extracted()


class MultiDocVectorStore:
    def __init__(self):
        self.docs = []

    def add_documents(self, document_id, texts, metadatas=None):
        for t, m in zip(texts, metadatas or []):
            self.docs.append({"document_id": document_id, "text": t, "metadata": m})

    def similarity_search(self, queries, document_id, k=2):
        return [d["text"] for d in self.docs if d["document_id"] == document_id]


class MultiDocLLM:
    def generate(self, prompt, **kwargs):
        if "Acetone" in prompt:
            return "language: English\njurisdiction: United States (OSHA / HazCom 2012)\ncompany name: ChemCorp\nproduct name: Acetone"
        elif "Ethanol" in prompt:
            return "language: English\njurisdiction: European Union (REACH / CLP)\ncompany name: BioFuel Inc\nproduct name: Ethanol"
        elif "Methanol" in prompt:
            return "language: Spanish\njurisdiction: Mexico (NOM-018-STPS)\ncompany name: MexiChem\nproduct name: Methanol"
        elif "Toluene" in prompt:
            return "language: German\njurisdiction: European Union (REACH / CLP)\ncompany name: Solvents AG\nproduct name: Toluene"
        else:
            return "language: English\njurisdiction: Canada (WHMIS 2015)\ncompany name: CanChem\nproduct name: Xylene"


@pytest.fixture
def temp_environment(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/test_5doc.db")
    db = SQLiteDatabase(settings)
    db.initialise()
    return db, settings


def test_five_document_workflow(temp_environment):
    db, settings = temp_environment
    repo = SQLiteDocumentRepository(db)
    vector_store = MultiDocVectorStore()

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=MultiDocReader(),
        text_cleaner=TextCleaner(),
        chunking_service=ChunkingService(settings),
        vector_store=vector_store,
        llm_client=MultiDocLLM(),
        metadata_validator=MetadataValidator(),
        settings=settings,
    )

    doc_names = ["Acetone", "Ethanol", "Methanol", "Toluene", "Xylene"]
    tmp_files = []

    try:
        results = []
        for name in doc_names:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(f"%PDF-1.4 Content for {name} SDS Section 1 Section 15".encode())
            tmp.close()
            tmp_path = Path(tmp.name)
            tmp_files.append(tmp_path)

            res = use_case.execute(tmp_path, f"{name.lower()}.pdf")
            results.append(res)

        # 1. Verify all 5 documents processed
        assert len(results) == 5
        for res in results:
            assert res.status == DocumentStatus.COMPLETED
            assert res.product_name in doc_names
            assert res.company_name is not None
            assert res.language is not None
            assert res.jurisdiction is not None

        # 2. Verify all 5 persisted separately in database
        all_db_docs = repo.find_all()
        assert len(all_db_docs) == 5

        # 3. Duplicate check on 1 document
        dup_res = use_case.execute(tmp_files[0], "acetone.pdf")
        assert dup_res.status == DocumentStatus.DUPLICATE
        assert dup_res.product_name == "Acetone"

    finally:
        for f in tmp_files:
            f.unlink(missing_ok=True)
