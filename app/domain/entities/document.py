"""Document entity — the core aggregate root of the platform.

An entity is identified by its ID and has a lifecycle (status transitions).
This class is intentionally framework-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.domain.enums.document_status import DocumentStatus
from app.domain.value_objects.sds_metadata import SDSMetadata


@dataclass
class Document:
    """Represents a processed document and its extraction results.

    This is the aggregate root — all state changes go through this class.
    """

    id: str
    filename: str
    file_path: str
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: SDSMetadata | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── State transitions ──────────────────────────────────────────────────────

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, metadata: SDSMetadata) -> None:
        self.status = DocumentStatus.COMPLETED
        self.metadata = metadata
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, reason: str) -> None:
        self.status = DocumentStatus.FAILED
        self.error_message = reason
        self.updated_at = datetime.now(timezone.utc)
