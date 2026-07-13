"""LLM response parser and metadata validator.

The LLM returns a deterministic 4-line format:
  language: <value>
  jurisdiction: <value>
  company name: <value>
  product name: <value>

This module:
  1. Parses the four expected lines from the LLM response.
  2. Normalises values (strip whitespace, handle "Unknown" → None, optional colons).
  3. Validates the parsed result with Pydantic.
  4. Returns a validated SDSMetadata value object.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, field_validator

from app.domain.exceptions.base import MetadataExtractionException
from app.domain.value_objects.sds_metadata import SDSMetadata

logger = logging.getLogger(__name__)

_UNKNOWN_VALUES = {"unknown", "n/a", "not available", "not found", "none", "-", ""}

# Robust field extraction patterns — label with optional colon, whitespace tolerant
_FIELD_PATTERNS: dict[str, re.Pattern] = {
    "language": re.compile(
        r"^\s*language\s*:?\s*(.+?)$", re.IGNORECASE | re.MULTILINE
    ),
    "jurisdiction": re.compile(
        r"^\s*jurisdiction\s*:?\s*(.+?)$", re.IGNORECASE | re.MULTILINE
    ),
    "company_name": re.compile(
        r"^\s*(?:company\s+name|company)\s*:?\s*(.+?)$", re.IGNORECASE | re.MULTILINE
    ),
    "product_name": re.compile(
        r"^\s*(?:product\s+name|product)\s*:?\s*(.+?)$", re.IGNORECASE | re.MULTILINE
    ),
}


class _ParsedMetadata(BaseModel):
    """Pydantic model for validating parsed LLM output fields."""

    language: str | None = None
    jurisdiction: str | None = None
    company_name: str | None = None
    product_name: str | None = None

    @field_validator("language", "jurisdiction", "company_name", "product_name", mode="before")
    @classmethod
    def normalise_value(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = str(v).strip()
        if cleaned.lower() in _UNKNOWN_VALUES:
            return None
        return cleaned if cleaned else None


class MetadataValidator:
    """Parses LLM text output and returns a validated SDSMetadata value object."""

    def parse_and_validate(self, file_id: str, llm_response: str) -> SDSMetadata:
        """Extract metadata fields from raw LLM text response.

        Expects the LLM to return 4 lines in format:
          language: <value>
          jurisdiction: <value>
          company name: <value>
          product name: <value>

        Args:
            file_id: The document ID to attach to the metadata.
            llm_response: Raw text returned by the LLM.

        Returns:
            SDSMetadata value object (fields may be None if not found).

        Raises:
            MetadataExtractionException: if parsing fails or all fields are empty.
        """
        logger.debug("Parsing LLM response | len=%d", len(llm_response))

        raw: dict[str, str | None] = {}
        
        # Extract all four fields using robust patterns
        for field_name, pattern in _FIELD_PATTERNS.items():
            match = pattern.search(llm_response)
            raw[field_name] = match.group(1).strip() if match else None

        # Validate that at least some fields were found
        if not any(raw.values()):
            logger.warning(
                "LLM response contained no parseable fields.\n\nFull Response:\n%s\n\nEnd Response",
                llm_response,
            )
            raise MetadataExtractionException(
                "LLM response did not contain any expected metadata fields. "
                "The model may have returned an unexpected format."
            )

        # Normalize and validate with Pydantic
        validated = _ParsedMetadata(**raw)

        result = SDSMetadata(
            file_id=file_id,
            language=validated.language,
            jurisdiction=validated.jurisdiction,
            company_name=validated.company_name,
            product_name=validated.product_name,
        )

        logger.info(
            "Metadata extracted | file_id=%s | lang=%s | jurisdiction=%s | company=%s | product=%s",
            file_id,
            result.language,
            result.jurisdiction,
            result.company_name,
            result.product_name,
        )
        return result
