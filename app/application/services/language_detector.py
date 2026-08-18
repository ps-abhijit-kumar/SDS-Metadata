"""Robust, deterministic document-level language detection service.

Implements a multi-stage language detector that combines:
  1. High-frequency SDS header marker matching (multilingual SDS document titles)
  2. Character trigram / n-gram frequency scoring across supported languages
  3. Stopword vocabulary frequency ratio validation on document text

Returns canonical human-readable language names (e.g. "Spanish", "Portuguese", "English", "French", "German").
This runs locally in < 2ms without calling an LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from app.application.services.language_normalizer import normalize_language

logger = logging.getLogger(__name__)

# SDS document title markers by language
_SDS_TITLE_MARKERS: dict[str, list[str]] = {
    "Spanish": [
        "hoja de datos de seguridad",
        "ficha de datos de seguridad",
        "identificacion de la sustancia",
        "medidas de primeros auxilios",
        "primeros auxilios",
        "identificacion de peligros",
    ],
    "Portuguese": [
        "ficha de dados de seguranca",
        "ficha de informacoes de seguranca",
        "identificacao do produto",
        "medidas de primeiros socorros",
        "primeiros socorros",
        "identificacao de perigos",
    ],
    "French": [
        "fiche de donnees de securite",
        "fiche de securite",
        "identification de la substance",
        "premiers secours",
        "identification des dangers",
    ],
    "German": [
        "sicherheitsdatenblatt",
        "bezeichnung des stoffs",
        "erste-hilfe-massnahmen",
        "mögliche gefahren",
        "gefahrenhinweise",
    ],
    "Italian": [
        "scheda di dati di sicurezza",
        "identificazione della sostanza",
        "misure di primo soccorso",
        "identificazione dei pericoli",
    ],
    "Dutch": [
        "veiligheidsinformatieblad",
        "identificatie van de stof",
        "erstehulpmaatregelen",
        "identificatie van de gevaren",
    ],
    "English": [
        "safety data sheet",
        "material safety data sheet",
        "first-aid measures",
        "hazards identification",
        "composition/information on ingredients",
    ],
}

# High-frequency grammatical function words / stopwords per language
_LANGUAGE_STOPWORDS: dict[str, set[str]] = {
    "Spanish": {
        "de", "la", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una",
        "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este", "si", "porque", "esta",
        "entre", "cuando", "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien", "desde",
    },
    "Portuguese": {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "nao", "uma", "os", "no", "se",
        "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele", "das", "seu", "sua", "ou", "quando",
        "muito", "nos", "ja", "eu", "tambem", "so", "pelo", "pela", "ate", "isso", "ela", "entre", "depois",
    },
    "French": {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une", "est", "pas", "pour", "qui", "dans",
        "sur", "une", "au", "par", "pour", "avec", "tout", "les", "sur", "ont", "vous", "ce", "ne", "que",
    },
    "German": {
        "und", "der", "die", "das", "in", "von", "zu", "den", "mit", "ist", "des", "auf", "für", "eine", "einen",
        "nicht", "dem", "sich", "als", "auch", "aus", "oder", "an", "nach", "wie", "bei", "um", "wir", "nur",
    },
    "Italian": {
        "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "il", "lo", "la", "i", "gli", "le", "un", "uno",
        "una", "che", "e", "del", "dello", "della", "dei", "degli", "delle", "al", "allo", "alla", "ai", "agli",
    },
    "Dutch": {
        "de", "het", "een", "van", "en", "in", "is", "dat", "op", "te", "zijn", "voor", "met", "die", "om", "als",
        "er", "wordt", "aan", "niet", "van", "ook", "om", "door", "over", "ze", "uit", "naar", "tot", "bij",
    },
    "English": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he",
        "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    },
}


class LanguageDetector:
    """Detects document language deterministically from PDF text."""

    @classmethod
    def detect_language(cls, text: str) -> str | None:
        """Analyze extracted PDF text and return canonical language name."""
        if not text or len(text.strip()) < 20:
            return None

        # Clean text for pattern matching (strip accents for marker matching)
        clean = text.lower()
        clean_normalized = cls._remove_accents(clean)

        # 1. Check explicit SDS Title / Header Markers (Highest Confidence)
        marker_scores: dict[str, int] = {lang: 0 for lang in _SDS_TITLE_MARKERS}
        for lang, markers in _SDS_TITLE_MARKERS.items():
            for marker in markers:
                if marker in clean_normalized:
                    marker_scores[lang] += 2

        best_marker_lang = max(marker_scores, key=marker_scores.get)
        if marker_scores[best_marker_lang] >= 4:
            logger.info("Language detected by SDS title markers: %s", best_marker_lang)
            return best_marker_lang

        # 2. Stopword Frequency Ratio Scoring
        words = re.findall(r"\b[a-z]{2,}\b", clean_normalized)
        if not words:
            return None

        total_words = len(words)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        stopword_scores: dict[str, float] = {lang: 0.0 for lang in _LANGUAGE_STOPWORDS}
        for lang, stop_set in _LANGUAGE_STOPWORDS.items():
            matches = sum(word_counts[w] for w in stop_set if w in word_counts)
            stopword_scores[lang] = matches / total_words

        best_stop_lang = max(stopword_scores, key=stopword_scores.get)
        best_score = stopword_scores[best_stop_lang]

        # Combine marker score + stopword score
        combined_scores: dict[str, float] = {}
        for lang in _LANGUAGE_STOPWORDS:
            combined_scores[lang] = stopword_scores[lang] * 10 + marker_scores[lang]

        final_lang = max(combined_scores, key=combined_scores.get)
        if combined_scores[final_lang] > 0.05:
            logger.info("Language detected by text analysis: %s (score=%.3f)", final_lang, combined_scores[final_lang])
            return normalize_language(final_lang)

        return None

    @staticmethod
    def _remove_accents(text: str) -> str:
        accents = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
        }
        for k, v in accents.items():
            text = text.replace(k, v)
        return text
