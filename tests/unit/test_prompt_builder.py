"""Unit tests for prompt_builder."""

from app.application.services.prompt_builder import build_extraction_prompt


def test_prompt_contains_required_field_labels():
    """Prompt should instruct the model to return the four required fields."""

    prompt = build_extraction_prompt(
        [
            "Product Name: TestChem",
            "Language: English",
            "Regulatory information",
        ]
    )

    lower = prompt.lower()

    assert "language:" in lower
    assert "jurisdiction:" in lower
    assert "company name:" in lower
    assert "product name:" in lower


def test_prompt_contains_context_chunks():
    """Retrieved chunks must appear inside the prompt."""

    chunks = [
        "chunk one",
        "chunk two",
        "chunk three",
    ]

    prompt = build_extraction_prompt(chunks)

    assert "chunk one" in prompt
    assert "chunk two" in prompt
    assert "chunk three" in prompt


def test_prompt_with_empty_chunks_uses_fallback():
    """Empty retrieval should still produce a valid prompt."""

    prompt = build_extraction_prompt([])

    assert "No document context available" in prompt


def test_prompt_requires_exactly_four_lines():
    """The prompt should force deterministic four-line output."""

    prompt = build_extraction_prompt(["context"]).lower()

    assert "exactly four" in prompt
    assert "four lines" in prompt


def test_prompt_instructs_no_fabrication():
    """Prompt should forbid hallucinated values."""

    prompt = build_extraction_prompt(["context"]).lower()

    assert "unknown" in prompt
    assert "cannot be determined" in prompt


def test_prompt_contains_language_jurisdiction_guidance():
    """Language and jurisdiction must be treated independently."""

    prompt = build_extraction_prompt(["context"]).lower()

    assert "language" in prompt
    assert "jurisdiction" in prompt
    assert "independent" in prompt


def test_prompt_is_compact():
    """
    Optimized prompt should remain compact.
    Prevent future prompt bloat.
    """

    prompt = build_extraction_prompt(["some context"])

    assert len(prompt) < 4000