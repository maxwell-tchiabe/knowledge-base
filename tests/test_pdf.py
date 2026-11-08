from pathlib import Path

import pymupdf as fitz

from knowledge_ingest.pdf import _native_page_text, extract_pages, metadata_from_pdf


def test_ocr_is_used_only_for_low_quality_pages(tmp_path: Path) -> None:
    source_root = tmp_path / "documents"
    source_root.mkdir()
    pdf_path = source_root / "mixed.pdf"

    with fitz.open() as document:
        native_page = document.new_page()
        native_page.insert_text(
            (72, 72),
            "This native page contains enough readable text to avoid OCR.",
        )
        document.new_page()
        document.save(pdf_path)

    ocr_calls: list[int] = []

    def fake_ocr(_image: bytes, page_number: int, _path: Path) -> str:
        ocr_calls.append(page_number)
        return "OCR text for the scanned page."

    pages, metadata = extract_pages(
        pdf_path, source_root, fake_ocr, minimum_native_characters=30
    )

    assert ocr_calls == [2]
    assert [page.used_ocr for page in pages] == [False, True]
    assert [page.page_number for page in pages] == [1, 2]
    assert metadata.title == "mixed"


def test_pdf_metadata_does_not_treat_creation_timestamp_as_publication_date(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "documents"
    source_root.mkdir()
    path = source_root / "law.pdf"

    metadata = metadata_from_pdf(
        path,
        source_root,
        {
            "title": "Law",
            "author": "Institution",
            "creationDate": "D:20250101000000",
        },
    )

    assert metadata.title == "Law"
    assert metadata.institution == "Institution"
    assert metadata.publication_date is None


def test_explicit_publication_date_is_parsed(tmp_path: Path) -> None:
    source_root = tmp_path / "documents"
    source_root.mkdir()
    path = source_root / "law.pdf"

    metadata = metadata_from_pdf(
        path,
        source_root,
        {"publicationDate": "2025-03-17"},
    )

    assert metadata.publication_date is not None
    assert metadata.publication_date.isoformat() == "2025-03-17"


def test_positioned_columns_are_preserved_as_a_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Kennzahl", fontsize=11)
        page.insert_text((333, 72), "2024", fontsize=11)
        page.insert_text((72, 90), "Preis", fontsize=11)
        page.insert_text((333, 90), "100", fontsize=11)
        document.save(pdf_path)

    with fitz.open(pdf_path) as document:
        extracted = _native_page_text(document[0])

    assert "| Kennzahl | 2024 |" in extracted
    assert "| Preis | 100 |" in extracted
