import pytest

from knowledge_ingest.models import PageContent
from knowledge_ingest.quality import (
    native_text_quality_issues,
    remove_repeated_page_edges,
    validate_page_sequence,
)


def test_sparse_and_suspicious_text_are_flagged() -> None:
    assert native_text_quality_issues("too short") == ("sparse_or_missing_text",)
    assert native_text_quality_issues("usable text\ufffd" * 10) == (
        "suspicious_characters",
    )
    assert (
        native_text_quality_issues(
            "A sufficiently long clean page of text with more than forty characters."
        )
        == ()
    )


def test_missing_page_is_reported() -> None:
    pages = (
        PageContent(1, "Page one", "", False),
        PageContent(3, "Page three", "", False),
    )

    with pytest.raises(ValueError, match="Page extraction mismatch"):
        validate_page_sequence(pages, expected_page_count=3)


def test_repeated_headers_and_footers_are_removed() -> None:
    pages = tuple(
        PageContent(number, f"Report title\nUnique section {number}\nPage footer", "", False)
        for number in range(1, 4)
    )

    cleaned = remove_repeated_page_edges(pages)

    assert all("Report title" not in text for text in cleaned)
    assert all("Page footer" not in text for text in cleaned)
    assert [f"Unique section {number}" in text for number, text in enumerate(cleaned, 1)] == [
        True,
        True,
        True,
    ]


def test_adjacent_duplicate_prose_is_removed_but_table_rows_are_retained() -> None:
    text = "Repeated sentence\nRepeated sentence\n| Value | 1 |\n| Value | 1 |"
    cleaned = remove_repeated_page_edges((PageContent(1, text, "", False),))

    assert cleaned == ("Repeated sentence\n| Value | 1 |\n| Value | 1 |",)
