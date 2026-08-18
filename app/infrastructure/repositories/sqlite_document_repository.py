"""SQLite implementation of the DocumentRepository interface.

Maps between the Document domain entity and flat SQLite rows.
No business logic lives here — only persistence concerns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.application.services.language_normalizer import normalize_language
from app.domain.entities.document import Document
from app.domain.enums.document_status import DocumentStatus
from app.domain.exceptions.base import DatabaseException, DocumentNotFoundException
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.sds_metadata import SDSMetadata
from app.infrastructure.database.sqlite_database import SQLiteDatabase

logger = logging.getLogger(__name__)


class SQLiteDocumentRepository(DocumentRepository):
    """Persists Document aggregates in a SQLite database."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._db = database

    # ── Write operations ───────────────────────────────────────────────────────

    def save(self, document: Document) -> None:
        sql = """
            INSERT INTO documents
                (id, filename, file_path, file_hash, status, product_name, company_name,
                 language, jurisdiction, processing_version, version_number, is_active,
                 error_message, processing_time_ms, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = self._to_row(document)
        try:
            with self._db.connection() as conn:
                conn.execute(sql, params)
            logger.debug("Saved document id=%s hash=%s ver=%d", document.id, document.file_hash, document.version_number)
        except Exception as exc:
            raise DatabaseException(f"Failed to save document {document.id}: {exc}") from exc

    def update(self, document: Document) -> None:
        sql = """
            UPDATE documents
               SET filename           = ?,
                   file_path          = ?,
                   file_hash          = ?,
                   status             = ?,
                   product_name       = ?,
                   company_name       = ?,
                   language           = ?,
                   jurisdiction       = ?,
                   processing_version = ?,
                   version_number     = ?,
                   is_active          = ?,
                   error_message      = ?,
                   processing_time_ms = ?,
                   updated_at         = ?
             WHERE id = ?
        """
        meta = document.metadata
        params = (
            document.filename,
            document.file_path,
            document.file_hash,
            document.status.value,
            meta.product_name if meta else None,
            meta.company_name if meta else None,
            meta.language if meta else None,
            meta.jurisdiction if meta else None,
            document.processing_version,
            document.version_number,
            1 if document.is_active else 0,
            document.error_message,
            document.processing_time_ms,
            document.updated_at.isoformat(),
            document.id,
        )
        try:
            with self._db.connection() as conn:
                cursor = conn.execute(sql, params)
                if cursor.rowcount == 0:
                    raise DocumentNotFoundException(f"Document not found: {document.id}")
            logger.debug("Updated document id=%s status=%s", document.id, document.status.value)
        except DocumentNotFoundException:
            raise
        except Exception as exc:
            raise DatabaseException(f"Failed to update document {document.id}: {exc}") from exc

    def delete(self, document_id: str) -> None:
        try:
            with self._db.connection() as conn:
                cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                if cursor.rowcount == 0:
                    raise DocumentNotFoundException(f"Document not found: {document_id}")
        except DocumentNotFoundException:
            raise
        except Exception as exc:
            raise DatabaseException(f"Failed to delete document {document_id}: {exc}") from exc

    # ── Read operations ────────────────────────────────────────────────────────

    def find_by_id(self, document_id: str) -> Document | None:
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            return self._to_entity(dict(row)) if row else None
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch document {document_id}: {exc}") from exc

    def find_by_hash(self, file_hash: str) -> Document | None:
        if not file_hash:
            return None
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE file_hash = ? AND status IN ('completed', 'duplicate') ORDER BY created_at DESC LIMIT 1",
                    (file_hash,)
                ).fetchone()
            return self._to_entity(dict(row)) if row else None
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch document by hash {file_hash}: {exc}") from exc

    def find_latest_by_filename(self, filename: str) -> Document | None:
        if not filename:
            return None
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE filename = ? AND status IN ('completed', 'duplicate') ORDER BY created_at DESC LIMIT 1",
                    (filename,)
                ).fetchone()
            return self._to_entity(dict(row)) if row else None
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch document by filename {filename}: {exc}") from exc

    def deactivate_previous_versions(self, filename: str, current_id: str) -> None:
        try:
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE documents SET is_active = 0 WHERE filename = ? AND id != ?",
                    (filename, current_id),
                )
        except Exception as exc:
            raise DatabaseException(f"Failed to deactivate previous versions for {filename}: {exc}") from exc

    def find_all(self) -> list[Document]:
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                ).fetchall()
            return [self._to_entity(dict(r)) for r in rows]
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch documents: {exc}") from exc

    def find_all_canonical(self) -> list[Document]:
        """Return unique active/canonical documents for chat operations."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE status IN ('completed', 'duplicate') AND is_active = 1 ORDER BY created_at DESC"
                ).fetchall()
            
            seen_keys = set()
            canonical_docs = []
            
            for r in rows:
                doc = self._to_entity(dict(r))
                key = doc.file_hash if doc.file_hash else doc.filename
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    canonical_docs.append(doc)
                    
            return canonical_docs
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch canonical documents: {exc}") from exc

    # ── Mapping helpers ────────────────────────────────────────────────────────

    def _to_row(self, doc: Document) -> tuple:
        meta = doc.metadata
        return (
            doc.id,
            doc.filename,
            doc.file_path,
            doc.file_hash,
            doc.status.value,
            meta.product_name if meta else None,
            meta.company_name if meta else None,
            normalize_language(meta.language) if meta else None,
            meta.jurisdiction if meta else None,
            doc.processing_version,
            doc.version_number,
            1 if doc.is_active else 0,
            doc.error_message,
            doc.processing_time_ms,
            doc.created_at.isoformat(),
            doc.updated_at.isoformat(),
        )

    def _to_entity(self, row: dict) -> Document:
        metadata = None
        if row.get("product_name") or row.get("company_name") or row.get("language") or row.get("jurisdiction"):
            metadata = SDSMetadata(
                file_id=row["id"],
                product_name=row.get("product_name"),
                company_name=row.get("company_name"),
                language=normalize_language(row.get("language")),
                jurisdiction=row.get("jurisdiction"),
            )

        return Document(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_hash=row.get("file_hash") or "",
            status=DocumentStatus(row["status"]),
            metadata=metadata,
            processing_version=row.get("processing_version") or "v1",
            version_number=row.get("version_number", 1) if row.get("version_number") is not None else 1,
            is_active=bool(row.get("is_active", 1)),
            error_message=row.get("error_message"),
            processing_time_ms=row.get("processing_time_ms"),
            created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
            updated_at=datetime.fromisoformat(row["updated_at"]).replace(tzinfo=timezone.utc),
        )
