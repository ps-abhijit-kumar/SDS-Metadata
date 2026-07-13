"""Text cleaning and normalisation utilities.

Called after PDF extraction and before chunking.
Removes artefacts introduced by PDF-to-text conversion without
destroying meaningful content.
"""

from __future__ import annotations

import re
import unicodedata


class TextCleaner:
    """Cleans raw PDF text for downstream processing."""

    # Regex patterns compiled once at class load
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _PAGE_NUMBER    = re.compile(r"(?m)^\s*\d+\s*$")
    _MULTIPLE_BLANK = re.compile(r"\n{3,}")
    _MULTIPLE_SPACE = re.compile(r" {2,}")
    _LIGATURES: dict[str, str] = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on raw PDF text."""
        if not text:
            return ""

        text = self._replace_ligatures(text)
        text = unicodedata.normalize("NFKC", text)
        text = self._CONTROL_CHARS.sub(" ", text)
        text = self._PAGE_NUMBER.sub("", text)
        text = self._MULTIPLE_SPACE.sub(" ", text)
        text = self._MULTIPLE_BLANK.sub("\n\n", text)
        return text.strip()

    def _replace_ligatures(self, text: str) -> str:
        for ligature, replacement in self._LIGATURES.items():
            text = text.replace(ligature, replacement)
        return text
