"""Data Transfer Objects for the extraction pipeline.

DTOs cross layer boundaries carrying data between application and presentation.
They are pure data containers — no business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.domain.enums.document_status import DocumentStatus


@dataclass(frozen=False)
class ExtractionResultDTO:
    """Final result returned to the presentation layer after processing."""

    document_id: str
    filename: str
    status: DocumentStatus
    product_name: str | None = None
    language: str | None = None
    jurisdiction: str | None = None
    company_name: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    debug_metadata: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        result = {
            "document_id": self.document_id,
            "filename": self.filename,
            "status": self.status.value,
            "product_name": self.product_name,
            "language": self.language,
            "jurisdiction": self.jurisdiction,
            "company_name": self.company_name,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if self.debug_metadata:
            result["debug_metadata"] = self.debug_metadata
        return result

