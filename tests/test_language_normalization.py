"""Tests for canonical language normalization."""

import pytest
from app.application.services.language_normalizer import normalize_language


def test_language_normalization_variants():
    assert normalize_language("es") == "Spanish"
    assert normalize_language("ES") == "Spanish"
    assert normalize_language("Spanish") == "Spanish"
    assert normalize_language("spanish") == "Spanish"
    assert normalize_language("español") == "Spanish"

    assert normalize_language("en") == "English"
    assert normalize_language("English") == "English"
    assert normalize_language("english") == "English"

    assert normalize_language("pt") == "Portuguese"
    assert normalize_language("de") == "German"
    assert normalize_language("fr") == "French"
    assert normalize_language("it") == "Italian"
    assert normalize_language("nl") == "Dutch"


def test_language_normalization_edge_cases():
    assert normalize_language(None) is None
    assert normalize_language("") is None
    assert normalize_language("  ") is None
