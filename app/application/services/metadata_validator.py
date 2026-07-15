"""LLM response parser and metadata validator.

The LLM returns a deterministic 4-line format:

    language: <value>
    jurisdiction: <value>
    company name: <value>
    product name: <value>

Responsibilities
----------------
1. Parse the four expected fields.
2. Normalize whitespace and unknown values.
3. Normalize jurisdiction names into canonical project values.
4. Validate the parsed result.
5. Return a production-ready SDSMetadata object.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, field_validator

from app.domain.exceptions.base import MetadataExtractionException
from app.domain.value_objects.sds_metadata import SDSMetadata

logger = logging.getLogger(__name__)

_UNKNOWN_VALUES = {
    "",
    "-",
    "none",
    "unknown",
    "n/a",
    "not available",
    "not found",
}

# ---------------------------------------------------------------------
# Robust field extraction patterns
# ---------------------------------------------------------------------

_FIELD_PATTERNS: dict[str, re.Pattern] = {
    "language": re.compile(
        r"^\s*language\s*:?\s*(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "jurisdiction": re.compile(
        r"^\s*jurisdiction\s*:?\s*(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "company_name": re.compile(
        r"^\s*(?:company\s+name|company)\s*:?\s*(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "product_name": re.compile(
        r"^\s*(?:product\s+name|product)\s*:?\s*(.+?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
}

# ---------------------------------------------------------------------
# Canonical jurisdiction mapping
# ---------------------------------------------------------------------

_JURISDICTION_MAP = {
    # United States
    "us": "United States (OSHA / HazCom 2012)",
    "usa": "United States (OSHA / HazCom 2012)",
    "united states": "United States (OSHA / HazCom 2012)",
    "united states of america": "United States (OSHA / HazCom 2012)",
    "osha": "United States (OSHA / HazCom 2012)",
    "hazcom": "United States (OSHA / HazCom 2012)",
    "hazcom 2012": "United States (OSHA / HazCom 2012)",

    # Canada
    "canada": "Canada (WHMIS 2015)",
    "whmis": "Canada (WHMIS 2015)",
    "whmis 2015": "Canada (WHMIS 2015)",

    # European Union
    "eu": "European Union (REACH / CLP)",
    "europe": "European Union (REACH / CLP)",
    "european union": "European Union (REACH / CLP)",
    "reach": "European Union (REACH / CLP)",
    "clp": "European Union (REACH / CLP)",

    # United Kingdom
    "uk": "United Kingdom (UK REACH)",
    "united kingdom": "United Kingdom (UK REACH)",
    "great britain": "United Kingdom (UK REACH)",
    "uk reach": "United Kingdom (UK REACH)",

    # Brazil
    "brazil": "Brazil (ABNT NBR 14725)",
    "brasil": "Brazil (ABNT NBR 14725)",
    "abnt": "Brazil (ABNT NBR 14725)",
    "abnt nbr 14725": "Brazil (ABNT NBR 14725)",

    # Australia
    "australia": "Australia (Safe Work Australia)",

    # New Zealand
    "new zealand": "New Zealand (HSNO)",
    "hsno": "New Zealand (HSNO)",

    # Japan
    "japan": "Japan (MHLW / JIS Z 7253)",

    # China
    "china": "China (GB/T 16483)",

    # South Korea
    "south korea": "South Korea (OSHACT K-REACH)",
    "k-reach": "South Korea (OSHACT K-REACH)",

    # India
    "india": "India (BIS / MSDS)",
    "bis": "India (BIS / MSDS)",

    # Singapore
    "singapore": "Singapore (WSH)",

    # Mexico
    "mexico": "Mexico (NOM-018-STPS)",
    "nom-018-stps": "Mexico (NOM-018-STPS)",
}


def _normalise_jurisdiction(value: str | None) -> str | None:
    """Convert LLM abbreviations into canonical project values."""

    if value is None:
        return None

    cleaned = value.strip()

    lookup = cleaned.lower()

    if lookup in _JURISDICTION_MAP:
        canonical = _JURISDICTION_MAP[lookup]

        if canonical != cleaned:
            logger.info(
                "Normalized jurisdiction: '%s' -> '%s'",
                cleaned,
                canonical,
            )

        return canonical

    return cleaned


class _ParsedMetadata(BaseModel):
    language: str | None = None
    jurisdiction: str | None = None
    company_name: str | None = None
    product_name: str | None = None

    @field_validator(
        "language",
        "jurisdiction",
        "company_name",
        "product_name",
        mode="before",
    )
    @classmethod
    def normalise_value(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        if cleaned.lower() in _UNKNOWN_VALUES:
            return None

        return cleaned or None


class MetadataValidator:
    """Parses and validates metadata returned by the LLM."""

    def parse_and_validate(
        self,
        file_id: str,
        llm_response: str,
    ) -> SDSMetadata:

        logger.debug(
            "Parsing LLM response | len=%d",
            len(llm_response),
        )

        raw: dict[str, str | None] = {}

        for field_name, pattern in _FIELD_PATTERNS.items():
            match = pattern.search(llm_response)
            raw[field_name] = match.group(1).strip() if match else None

        if not any(raw.values()):
            logger.warning(
                "No parseable metadata fields found.\n%s",
                llm_response,
            )
            raise MetadataExtractionException(
                "LLM response did not contain any expected metadata fields."
            )

        validated = _ParsedMetadata(**raw)

        metadata = SDSMetadata(
            file_id=file_id,
            language=validated.language,
            jurisdiction=_normalise_jurisdiction(validated.jurisdiction),
            company_name=validated.company_name,
            product_name=validated.product_name,
        )

        logger.info(
            "Metadata extracted | file_id=%s | lang=%s | jurisdiction=%s | company=%s | product=%s",
            file_id,
            metadata.language,
            metadata.jurisdiction,
            metadata.company_name,
            metadata.product_name,
        )

        return metadata