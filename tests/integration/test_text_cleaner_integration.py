"""Integration tests for the text cleaning pipeline."""

import pytest
from app.application.services.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_clean_realistic_pdf_text(cleaner):
    """Test cleaning realistic PDF-extracted text."""
    raw_text = """
    SAFETY DATA SHEET
    
    
    
    1 IDENTIFICATION
    
    
    Product Name: Aceton 99% (ACS)
    Manufacturer: ABC Chemical Corp
    
    
    
    
    2 HAZARDS IDENTIFICATION
    """
    result = cleaner.clean(raw_text)
    assert "SAFETY DATA SHEET" in result
    assert "Aceton 99%" in result
    assert "\n\n\n" not in result  # excessive newlines removed
