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

from app.application.services.language_normalizer import normalize_language
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
    "29 cfr 1910.1200": "United States (OSHA / HazCom 2012)",

    # Canada
    "canada": "Canada (WHMIS 2015)",
    "whmis": "Canada (WHMIS 2015)",
    "whmis 2015": "Canada (WHMIS 2015)",
    "canadian regulations": "Canada (WHMIS 2015)",
    "canada whmis 2015": "Canada (WHMIS 2015)",

    # European Union
    "eu": "European Union (REACH / CLP)",
    "europe": "European Union (REACH / CLP)",
    "european union": "European Union (REACH / CLP)",
    "reach": "European Union (REACH / CLP)",
    "clp": "European Union (REACH / CLP)",
    "reach / clp": "European Union (REACH / CLP)",

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
    "nbr 14725": "Brazil (ABNT NBR 14725)",
    "nbr 14725:2023": "Brazil (ABNT NBR 14725)",

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
    "nom-018-stps-2015": "Mexico (NOM-018-STPS)",
    "clasificado de acuerdo con nom-018-stps-2015": "Mexico (NOM-018-STPS)",
}


def _normalise_jurisdiction(value: str | None) -> str | None:
    """Convert LLM abbreviations or regulatory standard strings into canonical project values."""
    if value is None:
        return None

    cleaned = value.strip()
    lookup = cleaned.lower()

    if lookup in _JURISDICTION_MAP:
        return _JURISDICTION_MAP[lookup]

    # Partial substring matches for standard patterns
    if "whmis" in lookup:
        return "Canada (WHMIS 2015)"
    if "osha" in lookup or "hazcom" in lookup or "29 cfr" in lookup:
        return "United States (OSHA / HazCom 2012)"
    if "reach" in lookup or "clp" in lookup:
        return "European Union (REACH / CLP)"
    if "14725" in lookup or "abnt" in lookup:
        return "Brazil (ABNT NBR 14725)"
    if "nom-018" in lookup or "stps" in lookup:
        return "Mexico (NOM-018-STPS)"

    return cleaned


def _normalize_text_for_matching(text: str) -> str:
    """Normalization-aware string cleaner ignoring accents, whitespace, linebreaks, punctuation."""
    if not text:
        return ""
    text = text.lower()
    accents = {'á':'a','à':'a','ã':'a','â':'a','ä':'a','é':'e','è':'e','ê':'e','ë':'e','í':'i','ì':'i','î':'i','ï':'i','ó':'o','ò':'o','õ':'o','ô':'o','ö':'o','ú':'u','ù':'u','û':'u','ü':'u','ç':'c','ñ':'n'}
    for k, v in accents.items():
        text = text.replace(k, v)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


_FIELD_LABEL_BLACKLIST = {
    "identificador del producto", "nombre del producto", "nombre comercial", "product name",
    "trade name", "msds name", "name", "nombre", "company", "companhia", "company name",
    "manufacturer", "manufacturer name", "fabricante", "supplier", "distributor", "importer",
    "información sobre el fabricante", "información sobre el fabricante/importador/distribuidor",
    "informacion sobre el fabricante", "informacion sobre el fabricante/importador/distribuidor",
    "fornecedor", "ficha de dados", "ficha de datos", "datos de seguridad", "dados de segurança",
    "de dados de segurança", "emergency phone number", "teléfono de emergencia", "unknown", "n/a",
    "s or synonyms", "synonyms", "trade name or synonyms", "component", "hardener", "null", "none",
}

_GENERIC_LABEL_WORDS = {
    "importer", "distributor", "manufacturer", "supplier", "company", "companhia",
    "fabricante", "empresa", "fornecedor", "ficha", "dados", "seguranca", "seguridad",
}


def _is_invalid_field_value(val: str | None) -> bool:
    """Check if a string is a structural label rather than a real metadata value."""
    if not val or not val.strip():
        return True
    cleaned = val.strip().lower()
    if cleaned in _FIELD_LABEL_BLACKLIST:
        return True
    if cleaned in _GENERIC_LABEL_WORDS:
        return True
    if cleaned.startswith("de dados") or cleaned.startswith("identificador del"):
        return True
    return False


def _verify_and_disambiguate_company(company_val: str | None, context_text: str) -> str | None:
    """Disambiguate Manufacturer vs Brand/Supplier using structural document context."""
    if _is_invalid_field_value(company_val):
        company_val = None

    norm_context = _normalize_text_for_matching(context_text)

    # If LLM extracted a valid company name grounded in text, preserve it unless explicitly invalid
    if company_val:
        norm_val = _normalize_text_for_matching(company_val)
        if norm_val and norm_val in norm_context:
            logger.info("Preserving grounded LLM company name: '%s'", company_val)
            return company_val

    # Check explicit Company / Companhia / Manufacturer / Fabricante header in text
    comp_match = re.search(
        r"(?:fornecedor\s+da\s+ficha\s+de\s+dados\s+de\s+segurança|companhia|company\s+name|company|manufacturer\s+name|manufacturer|fabricante|importer\s*/?\s*distributor)\s*:?\s*([^\n\r]{3,100})",
        context_text,
        re.IGNORECASE,
    )
    if comp_match:
        explicit_company = comp_match.group(1).strip()
        explicit_company = re.split(r"[\n\r]", explicit_company)[0].strip()
        # Clean leading header words if present
        explicit_company = re.sub(r"^(?:distributor|importer|supplier|manufacturer|company|name|nombre|fabricante)\s*:?\s*", "", explicit_company, flags=re.IGNORECASE).strip()
        if len(explicit_company) > 3 and not _is_invalid_field_value(explicit_company):
            logger.info("Structural Disambiguation: Resolved company '%s' -> '%s'", company_val, explicit_company)
            return explicit_company

    return company_val


def _verify_and_disambiguate_product(product_val: str | None, context_text: str) -> str | None:
    """Disambiguate SDS Product Name vs Kit Component / Ingredient Name."""
    if _is_invalid_field_value(product_val):
        product_val = None

    prod_match = re.search(
        r"(?:product\s+name(?:\(s\))?(?:\s+or\s+synonyms)?|nome\s+do\s+produto|msds\s+name|identificador\s+del\s+producto|trade\s+name)\s*:?\s*([^\n\r]{3,150})",
        context_text,
        re.IGNORECASE,
    )
    if prod_match:
        explicit_product = prod_match.group(1).strip()
        explicit_product = re.split(r"[\n\r]", explicit_product)[0].strip()
        explicit_product = re.sub(r"^(?:\(s\)|or\s+synonyms|:\s*)+", "", explicit_product, flags=re.IGNORECASE).strip()
        if "hardener" in (product_val or "").lower() and "hardener" not in explicit_product.lower():
            logger.info("Structural Disambiguation: Overriding component '%s' with explicit SDS Product Name '%s'", product_val, explicit_product)
            return explicit_product
        if len(explicit_product) > 2 and not _is_invalid_field_value(explicit_product):
            return explicit_product

    return product_val


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
    """Parses, disambiguates, and validates metadata returned by the LLM."""

    def parse_and_validate(
        self,
        file_id: str,
        llm_response: str,
        context_text: str = "",
    ) -> SDSMetadata:

        logger.debug("Parsing LLM response | len=%d", len(llm_response))

        raw: dict[str, str | None] = {}

        for field_name, pattern in _FIELD_PATTERNS.items():
            match = pattern.search(llm_response)
            raw[field_name] = match.group(1).strip() if match else None

        if not any(raw.values()):
            logger.warning("No parseable metadata fields found.\n%s", llm_response)
            raise MetadataExtractionException("LLM response did not contain any expected metadata fields.")

        validated = _ParsedMetadata(**raw)

        company = _verify_and_disambiguate_company(validated.company_name, context_text) if context_text else validated.company_name
        product = _verify_and_disambiguate_product(validated.product_name, context_text) if context_text else validated.product_name

        metadata = SDSMetadata(
            file_id=file_id,
            language=normalize_language(validated.language),
            jurisdiction=_normalise_jurisdiction(validated.jurisdiction),
            company_name=company,
            product_name=product,
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