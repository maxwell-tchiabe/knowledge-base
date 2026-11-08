import tiktoken

from knowledge_ingest.chunking import build_chunks, content_units
from knowledge_ingest.models import PageContent


def test_chunking_respects_token_limit_and_carries_page_references() -> None:
    text = " ".join(
        f"Immobilienfinanzierung Zahl {number} beschreibt den Markt."
        for number in range(100)
    )
    pages = (
        PageContent(1, text, text, False),
        PageContent(2, "Zusätzlicher Absatz mit weiteren Informationen.", "", False),
    )

    chunks = build_chunks(pages, chunk_size_tokens=40, chunk_overlap_tokens=8)

    assert len(chunks) > 2
    assert all(chunk.page_numbers for chunk in chunks)
    assert {1, 2}.issubset({page for chunk in chunks for page in chunk.page_numbers})
    encoder = tiktoken.get_encoding("cl100k_base")
    assert all(len(encoder.encode(chunk.content)) <= 40 for chunk in chunks)


def test_markdown_table_is_kept_as_one_content_unit() -> None:
    table = "| Kennzahl | 2024 | 2025 |\n| --- | --- | --- |\n| Preis | 100 | 120 |"
    page = PageContent(1, table, table, False)

    units = content_units((page,))
    chunks = build_chunks((page,), chunk_size_tokens=5, chunk_overlap_tokens=1)

    assert len(units) == 1
    assert units[0].is_table
    assert len(chunks) == 1
    assert chunks[0].content == table


def test_section_heading_is_recorded_and_starts_a_new_chunk() -> None:
    pages = (
        PageContent(1, "1 Einleitung\n\nErster Inhalt.", "", False),
        PageContent(2, "2 Ergebnisse\n\nZweiter Inhalt.", "", False),
    )

    chunks = build_chunks(pages, chunk_size_tokens=30, chunk_overlap_tokens=3)

    assert [chunk.section_title for chunk in chunks] == [
        "1 Einleitung",
        "2 Ergebnisse",
    ]
