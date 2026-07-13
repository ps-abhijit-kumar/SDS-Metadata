"""SQLite implementation of the DocumentRepository interface.

Maps between the Document domain entity and flat SQLite rows.
No business logic lives here — only persistence concerns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
                (id, filename, file_path, status, product_name, language,
                 jurisdiction, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = self._to_row(document)
        try:
            with self._db.connection() as conn:
                conn.execute(sql, params)
            logger.debug("Saved document id=%s", document.id)
        except Exception as exc:
            raise DatabaseException(f"Failed to save document {document.id}: {exc}") from exc

    def update(self, document: Document) -> None:
        sql = """
            UPDATE documents
               SET filename      = ?,
                   file_path     = ?,
                   status        = ?,
                   product_name  = ?,
                   language      = ?,
                   jurisdiction  = ?,
                   error_message = ?,
                   updated_at    = ?
             WHERE id = ?
        """
        meta = document.metadata
        params = (
            document.filename,
            document.file_path,
            document.status.value,
            meta.product_name if meta else None,
            meta.language if meta else None,
            meta.jurisdiction if meta else None,
            document.error_message,
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

    def find_all(self) -> list[Document]:
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                ).fetchall()
            return [self._to_entity(dict(r)) for r in rows]
        except Exception as exc:
            raise DatabaseException(f"Failed to fetch documents: {exc}") from exc

    # ── Mapping helpers ────────────────────────────────────────────────────────

    def _to_row(self, doc: Document) -> tuple:
        meta = doc.metadata
        return (
            doc.id,
            doc.filename,
            doc.file_path,
            doc.status.value,
            meta.product_name if meta else None,
            meta.language if meta else None,
            meta.jurisdiction if meta else None,
            doc.error_message,
            doc.created_at.isoformat(),
            doc.updated_at.isoformat(),
        )

    def _to_entity(self, row: dict) -> Document:
        metadata = None
        if row.get("product_name") or row.get("language") or row.get("jurisdiction"):
            metadata = SDSMetadata(
                file_id=row["id"],
                product_name=row.get("product_name"),
                language=row.get("language"),
                jurisdiction=row.get("jurisdiction"),
            )

        return Document(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            status=DocumentStatus(row["status"]),
            metadata=metadata,
            error_message=row.get("error_message"),
            created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
            updated_at=datetime.fromisoformat(row["updated_at"]).replace(tzinfo=timezone.utc),
        )
