"""Unit tests for MetadataValidator."""

import pytest
from app.application.services.metadata_validator import MetadataValidator
from app.domain.exceptions.base import MetadataExtractionException


@pytest.fixture
def validator() -> MetadataValidator:
    return MetadataValidator()


def test_parse_all_four_fields(validator):
    response = (
        "Product Name: Aceton 99%\n"
        "Company Name: Sigma-Aldrich\n"
        "Language: English\n"
        "Jurisdiction: United States (OSHA / HazCom 2012)"

    )
    result = validator.parse_and_validate("file-001", response)
    assert result.file_id == "file-001"
    assert result.product_name == "Aceton 99%"
    assert result.company_name == "Sigma-Aldrich"
    assert result.language == "English"
    assert result.jurisdiction == "United States (OSHA / HazCom 2012)"


def test_parse_strips_whitespace(validator):
    response = (
        "Product Name:   Trimethylbenzene  \n"
        "Language:  Portuguese  \n"
        "Jurisdiction:   Brazil (ABNT NBR 14725)  "
    )
    result = validator.parse_and_validate("file-002", response)
    assert result.product_name == "Trimethylbenzene"
    assert result.language == "Portuguese"
    assert result.jurisdiction == "Brazil (ABNT NBR 14725)"


def test_parse_unknown_values_become_none(validator):
    response = (
        "Product Name: Unknown\n"
        "Language: English\n"
        "Jurisdiction: N/A"
    )
    result = validator.parse_and_validate("file-003", response)
    assert result.product_name is None
    assert result.language == "English"
    assert result.jurisdiction is None


def test_parse_partial_fields_succeeds(validator):
    response = (
        "Product Name: Hydrochloric Acid\n"
        "Language: German"
    )
    result = validator.parse_and_validate("file-004", response)
    assert result.product_name == "Hydrochloric Acid"
    assert result.language == "German"
    assert result.jurisdiction is None


def test_parse_no_fields_raises_exception(validator):
    with pytest.raises(MetadataExtractionException):
        validator.parse_and_validate("file-005", "This is just some random text with no fields.")


def test_parse_case_insensitive_labels(validator):
    response = (
        "PRODUCT NAME: Toluene\n"
        "LANGUAGE: Spanish\n"
        "JURISDICTION: Mexico (NOM-018-STPS)"
    )
    result = validator.parse_and_validate("file-006", response)
    assert result.product_name == "Toluene"
    assert result.language == "Spanish"
    assert result.jurisdiction == "Mexico (NOM-018-STPS)"


def test_parse_is_complete_true(validator):
    response = (
        "Product Name: Benzene\n"
        "Company Name: Sigma-Aldrich\n"
        "Language: English\n"
        "Jurisdiction: European Union (REACH / CLP)"
    )
    result = validator.parse_and_validate("file-007", response)
    assert result.is_complete() is True


def test_parse_is_complete_false_missing_field(validator):
    response = (
        "Product Name: Benzene\n"
        "Language: English\n"
        "Jurisdiction: Unknown"
    )
    result = validator.parse_and_validate("file-008", response)
    assert result.is_complete() is False
