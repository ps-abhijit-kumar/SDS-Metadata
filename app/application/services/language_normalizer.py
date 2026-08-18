"""Canonical language normalization service."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Map common ISO 639-1 language codes, native names, and variants to canonical names
_LANGUAGE_MAP: dict[str, str] = {
    # English
    "en": "English",
    "eng": "English",
    "english": "English",
    "inglés": "English",
    "ingles": "English",
    "englisch": "English",
    "anglais": "English",
    "inglês": "English",

    # Spanish
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "castellano": "Spanish",

    # Portuguese
    "pt": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "portugues": "Portuguese",

    # German
    "de": "German",
    "ger": "German",
    "deu": "German",
    "german": "German",
    "deutsch": "German",
    "alemán": "German",
    "aleman": "German",

    # French
    "fr": "French",
    "fre": "French",
    "fra": "French",
    "french": "French",
    "français": "French",
    "francais": "French",
    "francés": "French",
    "frances": "French",

    # Italian
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "italiano": "Italian",

    # Dutch
    "nl": "Dutch",
    "dut": "Dutch",
    "nld": "Dutch",
    "dutch": "Dutch",
    "nederlands": "Dutch",

    # Swedish
    "sv": "Swedish",
    "swe": "Swedish",
    "swedish": "Swedish",
    "svenska": "Swedish",

    # Danish
    "da": "Danish",
    "dan": "Danish",
    "danish": "Danish",
    "dansk": "Danish",

    # Norwegian
    "no": "Norwegian",
    "nor": "Norwegian",
    "norwegian": "Norwegian",
    "norsk": "Norwegian",

    # Polish
    "pl": "Polish",
    "pol": "Polish",
    "polish": "Polish",
    "polski": "Polish",

    # Chinese
    "zh": "Chinese",
    "chi": "Chinese",
    "zho": "Chinese",
    "chinese": "Chinese",

    # Japanese
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
}


def normalize_language(value: str | None) -> str | None:
    """Convert language code or name into canonical human-readable string.

    Examples:
        normalize_language("es") -> "Spanish"
        normalize_language("Spanish") -> "Spanish"
        normalize_language("ES") -> "Spanish"
        normalize_language("en") -> "English"
    """
    if not value:
        return None

    cleaned = str(value).strip()
    if not cleaned:
        return None

    lookup = cleaned.lower()

    if lookup in _LANGUAGE_MAP:
        canonical = _LANGUAGE_MAP[lookup]
        if canonical != cleaned:
            logger.info("Normalized language: '%s' -> '%s'", cleaned, canonical)
        return canonical

    # If already capitalized multi-character word not in dict, title case it
    return cleaned.capitalize() if len(cleaned) > 3 else cleaned.upper()
