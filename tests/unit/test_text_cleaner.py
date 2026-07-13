"""Unit tests for TextCleaner."""

import pytest
from app.application.services.text_cleaner import TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


def test_clean_empty_string_returns_empty(cleaner):
    assert cleaner.clean("") == ""


def test_clean_removes_control_characters(cleaner):
    result = cleaner.clean("Hello\x00World\x01test")
    assert "\x00" not in result
    assert "\x01" not in result
    assert "Hello" in result
    assert "World" in result


def test_clean_normalises_multiple_spaces(cleaner):
    result = cleaner.clean("Hello     World")
    assert "  " not in result


def test_clean_normalises_excessive_newlines(cleaner):
    result = cleaner.clean("Line1\n\n\n\n\nLine2")
    assert "\n\n\n" not in result


def test_clean_removes_standalone_page_numbers(cleaner):
    text = "Section 1\n\n42\n\nSome content"
    result = cleaner.clean(text)
    lines = [l.strip() for l in result.splitlines()]
    assert "42" not in lines


def test_clean_replaces_ligatures(cleaner):
    result = cleaner.clean("\ufb01nal\ufb02ame")  # fi, fl ligatures
    assert "fi" in result
    assert "fl" in result
    assert "\ufb01" not in result
    assert "\ufb02" not in result


def test_clean_full_pipeline(cleaner):
    raw = "  SAFETY DATA SHEET\x00\n\n\n\nProduct Name:  \ufb01ne Chemical   \n\n1\n\nSection 1\n"
    result = cleaner.clean(raw)
    assert "SAFETY DATA SHEET" in result
    assert "fine Chemical" in result
    assert "\x00" not in result
