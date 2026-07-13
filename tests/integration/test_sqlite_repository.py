"""Integration tests for SQLiteDocumentRepository."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from app.domain.entities.document import Document
from app.domain.enums.document_status import DocumentStatus
from app.domain.exceptions.base import DocumentNotFoundException, DatabaseException
from app.domain.value_objects.sds_metadata import SDSMetadata
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.configuration.settings import Settings


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        settings = type('Settings', (), {
            'db_path': db_path,
            'database_url': f'sqlite:///{db_path}'
        })()
        db = SQLiteDatabase(settings)
        db.initialise()
        yield db


@pytest.fixture
def repository(temp_db):
    """Create a repository with the temporary database."""
    return SQLiteDocumentRepository(temp_db)


def test_save_and_retrieve(repository):
    """Test saving and retrieving a document."""
    doc = Document(
        id="doc-001",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        status=DocumentStatus.PENDING,
    )
    repository.save(doc)
    retrieved = repository.find_by_id("doc-001")
    assert retrieved is not None
    assert retrieved.filename == "test.pdf"


def test_find_nonexistent_returns_none(repository):
    """Test that finding a nonexistent document returns None."""
    result = repository.find_by_id("nonexistent")
    assert result is None


def test_update_document(repository):
    """Test updating a document record."""
    doc = Document(
        id="doc-002",
        filename="test2.pdf",
        file_path="/tmp/test2.pdf",
    )
    repository.save(doc)

    meta = SDSMetadata(
        file_id="doc-002",
        product_name="Test Product",
        language="English",
        jurisdiction="US",
    )
    doc.mark_completed(meta)
    repository.update(doc)

    retrieved = repository.find_by_id("doc-002")
    assert retrieved.metadata is not None
    assert retrieved.metadata.product_name == "Test Product"


def test_update_nonexistent_raises(repository):
    """Test that updating a nonexistent document raises."""
    doc = Document(id="nonexistent", filename="x.pdf", file_path="/tmp/x.pdf")
    with pytest.raises(DocumentNotFoundException):
        repository.update(doc)


def test_find_all_returns_all_documents(repository):
    """Test that find_all returns all stored documents."""
    for i in range(3):
        doc = Document(
            id=f"doc-{i:03d}",
            filename=f"test{i}.pdf",
            file_path=f"/tmp/test{i}.pdf",
        )
        repository.save(doc)

    all_docs = repository.find_all()
    assert len(all_docs) == 3


def test_delete_document(repository):
    """Test deleting a document."""
    doc = Document(id="doc-del", filename="x.pdf", file_path="/tmp/x.pdf")
    repository.save(doc)
    assert repository.find_by_id("doc-del") is not None

    repository.delete("doc-del")
    assert repository.find_by_id("doc-del") is None


def test_delete_nonexistent_raises(repository):
    """Test that deleting a nonexistent document raises."""
    with pytest.raises(DocumentNotFoundException):
        repository.delete("nonexistent")
