from __future__ import annotations

import re

import tiktoken

from knowledge_ingest.models import Chunk, ContentUnit, PageContent
from knowledge_ingest.quality import remove_repeated_page_edges

_HEADING = re.compile(
    r"^\s*(?:(?:§{1,2})\s*\d+[a-z]?(?:\.\d+)*|"
    r"(?:\d+(?:\.\d+)*\.?\s+[A-ZÄÖÜ].*)|"
    r"(?i:Kapitel|Abschnitt|Teil)\s+\S+|"
    r"[A-ZÄÖÜ][A-ZÄÖÜ0-9 ,;:()/-]{4,})\s*$",
)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def _is_heading(line: str) -> bool:
    return bool(_HEADING.match(line.strip())) and len(line.strip()) < 160


def content_units(pages: tuple[PageContent, ...]) -> tuple[ContentUnit, ...]:
    cleaned_pages = remove_repeated_page_edges(pages)
    units: list[ContentUnit] = []
    current_section: str | None = None

    for page, page_text in zip(pages, cleaned_pages, strict=True):
        lines = page_text.splitlines()
        paragraph: list[str] = []

        def flush_paragraph() -> None:
            text = " ".join(line.strip() for line in paragraph if line.strip()).strip()
            if text:
                units.append(
                    ContentUnit(text, page.page_number, current_section)
                )
            paragraph.clear()

        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                flush_paragraph()
                index += 1
                continue
            if _TABLE_LINE.match(line):
                flush_paragraph()
                table_lines: list[str] = []
                while index < len(lines) and _TABLE_LINE.match(lines[index].strip()):
                    table_lines.append(lines[index].strip())
                    index += 1
                units.append(
                    ContentUnit(
                        "\n".join(table_lines),
                        page.page_number,
                        current_section,
                        is_table=True,
                    )
                )
                continue
            if _is_heading(line):
                flush_paragraph()
                current_section = line
                index += 1
                continue
            paragraph.append(line)
            index += 1
        flush_paragraph()
    return tuple(units)


def _tokens(text: str, encoder: tiktoken.Encoding) -> list[int]:
    return encoder.encode(text)


def _split_oversized_paragraph(
    unit: ContentUnit,
    limit: int,
    overlap_tokens: int,
    encoder: tiktoken.Encoding,
) -> list[ContentUnit]:
    token_ids = _tokens(unit.text, encoder)
    if unit.is_table or len(token_ids) <= limit:
        return [unit]
    piece_size = max(1, limit - overlap_tokens - 2)
    return [
        ContentUnit(
            encoder.decode(token_ids[start : start + piece_size]),
            unit.page_number,
            unit.section_title,
        )
        for start in range(0, len(token_ids), piece_size)
    ]


def _overlap_segments(
    segments: list[ContentUnit], overlap_tokens: int, encoder: tiktoken.Encoding
) -> list[ContentUnit]:
    if overlap_tokens == 0:
        return []
    overlap: list[ContentUnit] = []
    remaining = overlap_tokens
    for segment in reversed(segments):
        if segment.is_table:
            continue
        token_ids = _tokens(segment.text, encoder)
        if len(token_ids) <= remaining:
            overlap.append(segment)
            remaining -= len(token_ids)
        else:
            overlap.append(
                ContentUnit(
                    encoder.decode(token_ids[-remaining:]),
                    segment.page_number,
                    segment.section_title,
                )
            )
            break
        if remaining == 0:
            break
    return list(reversed(overlap))


def build_chunks(
    pages: tuple[PageContent, ...],
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> tuple[Chunk, ...]:
    if chunk_size_tokens < 1:
        raise ValueError("chunk_size_tokens must be greater than zero.")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            "chunk_overlap_tokens must be at least zero and smaller than chunk_size_tokens."
        )

    encoder = tiktoken.get_encoding("cl100k_base")
    units = [
        split
        for unit in content_units(pages)
        for split in _split_oversized_paragraph(
            unit, chunk_size_tokens, chunk_overlap_tokens, encoder
        )
    ]
    chunks: list[Chunk] = []
    current: list[ContentUnit] = []
    current_section: str | None = None

    def emit(segments: list[ContentUnit], section: str | None) -> None:
        if not segments:
            return
        text = "\n\n".join(segment.text for segment in segments).strip()
        if not text:
            return
        page_numbers = tuple(sorted({segment.page_number for segment in segments}))
        chunks.append(
            Chunk(len(chunks), text, page_numbers, section)
        )

    for unit in units:
        if current and unit.section_title != current_section:
            emit(current, current_section)
            current = []
        if not current:
            current_section = unit.section_title
        candidate = "\n\n".join(
            [segment.text for segment in current] + [unit.text]
        )
        if current and len(_tokens(candidate, encoder)) > chunk_size_tokens:
            previous = current
            emit(previous, current_section)
            current = _overlap_segments(
                previous, chunk_overlap_tokens, encoder
            )
            if current and unit.section_title != current_section:
                current = []
            while current:
                candidate = "\n\n".join(
                    [segment.text for segment in current] + [unit.text]
                )
                if len(_tokens(candidate, encoder)) <= chunk_size_tokens:
                    break
                if len(current) > 1:
                    current.pop(0)
                    continue
                overlap_tokens = _tokens(current[0].text, encoder)
                overflow = len(_tokens(candidate, encoder)) - chunk_size_tokens
                if overflow >= len(overlap_tokens):
                    current = []
                else:
                    current[0] = ContentUnit(
                        encoder.decode(overlap_tokens[overflow:]),
                        current[0].page_number,
                        current[0].section_title,
                    )
            current.append(unit)
        else:
            current.append(unit)

    emit(current, current_section)
    return tuple(chunks)
