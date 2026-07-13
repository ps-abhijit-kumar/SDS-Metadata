"""SDS document metadata value object.

A value object is immutable and identified by its content, not by an ID.
This represents the extracted metadata result for one SDS document.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SDSMetadata:
    """Extracted metadata for a single Safety Data Sheet.

    All fields are optional because extraction may partially succeed.
    None indicates the field could not be confidently determined.
    """

    file_id: str
    product_name: str | None = None
    language: str | None = None
    jurisdiction: str | None = None
    company_name: str | None = None

    def is_complete(self) -> bool:
        """Return True only when all four fields were successfully extracted."""
        return all([
            self.product_name,
            self.language,
            self.jurisdiction,
            self.company_name,
        ])

    def to_dict(self) -> dict[str, str | None]:
        return {
            "file_id": self.file_id,
            "product_name": self.product_name,
            "language": self.language,
            "jurisdiction": self.jurisdiction,
            "company_name": self.company_name,
        }
