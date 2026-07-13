"""Unit tests for the prompt builder."""

from app.application.services.prompt_builder import build_extraction_prompt


def test_prompt_contains_required_field_labels():
    chunks = ["Product Name: TestChem", "Language: English", "Regulatory info"]
    prompt = build_extraction_prompt(chunks)
    assert "Product Name:" in prompt
    assert "Language:" in prompt
    assert "Jurisdiction:" in prompt


def test_prompt_contains_context_chunks():
    chunks = ["This is a safety data sheet.", "Section 1 identification."]
    prompt = build_extraction_prompt(chunks)
    assert "This is a safety data sheet." in prompt
    assert "Section 1 identification." in prompt


def test_prompt_with_empty_chunks_uses_fallback():
    prompt = build_extraction_prompt([])
    assert "No document context available" in prompt


def test_prompt_contains_jurisdiction_examples():
    prompt = build_extraction_prompt(["some context"])
    assert "OSHA" in prompt
    assert "REACH" in prompt
    assert "ABNT NBR 14725" in prompt
    assert "WHMIS" in prompt


def test_prompt_instructs_no_fabrication():
    prompt = build_extraction_prompt(["context"])
    assert "Unknown" in prompt


def test_prompt_separates_chunks():
    chunks = ["chunk one", "chunk two", "chunk three"]
    prompt = build_extraction_prompt(chunks)
    assert "chunk one" in prompt
    assert "chunk two" in prompt
    assert "chunk three" in prompt


def test_prompt_language_jurisdiction_independence():
    prompt = build_extraction_prompt(["some sds content"])
    # The prompt must explicitly note that language and jurisdiction are independent
    assert "INDEPENDENT" in prompt.upper() or "independent" in prompt.lower()
