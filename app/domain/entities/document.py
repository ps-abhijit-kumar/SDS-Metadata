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
    file_hash: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: SDSMetadata | None = None
    processing_version: str = "v3"
    version_number: int = 1
    is_active: bool = True
    error_message: str | None = None
    processing_time_ms: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── State transitions ──────────────────────────────────────────────────────

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, metadata: SDSMetadata, processing_time_ms: float | None = None) -> None:
        self.status = DocumentStatus.COMPLETED
        self.metadata = metadata
        self.processing_time_ms = processing_time_ms
        self.updated_at = datetime.now(timezone.utc)

    def mark_duplicate(self, existing_metadata: SDSMetadata) -> None:
        self.status = DocumentStatus.DUPLICATE
        self.metadata = existing_metadata
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, reason: str) -> None:
        self.status = DocumentStatus.FAILED
        self.error_message = reason
        self.updated_at = datetime.now(timezone.utc)
