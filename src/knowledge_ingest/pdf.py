from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import median

import pymupdf as fitz

from knowledge_ingest.models import DocumentMetadata, PageContent
from knowledge_ingest.quality import native_text_quality_issues, validate_page_sequence

OCRCallback = Callable[[bytes, int, Path], str]


@dataclass
class _PositionedLine:
    baseline: float
    y0: float
    y1: float
    words: list[tuple[float, float, str]] = field(default_factory=list)


def _format_table(rows: list[list[str | None]]) -> str:
    normalized = [
        [cell.replace("|", r"\|").replace("\n", " ").strip() if cell else "" for cell in row]
        for row in rows
    ]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    lines = ["| " + " | ".join(normalized[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _native_page_text(page: fitz.Page) -> str:
    words = page.get_text("words", sort=True)
    if not words:
        return ""

    line_height = median(word[3] - word[1] for word in words)
    line_tolerance = max(3.0, line_height * 0.45)
    column_gap = max(8.0, line_height * 1.8)
    word_lines: list[_PositionedLine] = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        x0, y0, x1, y1, text = word[:5]
        if (
            not word_lines
            or abs(y0 - word_lines[-1].baseline) > line_tolerance
        ):
            word_lines.append(_PositionedLine(y0, y0, y1))
        line = word_lines[-1]
        line.y0 = min(line.y0, y0)
        line.y1 = max(line.y1, y1)
        line.words.append((x0, x1, text))

    lines: list[tuple[float, float, list[str]]] = []
    for positioned_line in word_lines:
        positioned_line.words.sort(key=lambda word: word[0])
        cells: list[str] = []
        previous_x1: float | None = None
        for word in positioned_line.words:
            x0, x1, text = word
            if previous_x1 is not None and x0 - previous_x1 > column_gap:
                cells.append(text)
            elif cells:
                cells[-1] = f"{cells[-1]} {text}"
            else:
                cells.append(text)
            previous_x1 = x1
        lines.append(
            (positioned_line.y0, positioned_line.y1, cells)
        )
    lines.sort(key=lambda line: line[0])

    positioned: list[tuple[float, float, str]] = []
    index = 0
    table_line_gap = max(14.0, line_height * 2.5)
    while index < len(lines):
        y0, y1, cells = lines[index]
        if len(cells) < 2:
            positioned.append((y0, y1, cells[0]))
            index += 1
            continue

        table_rows = [cells]
        next_index = index + 1
        while next_index < len(lines):
            next_y0, _, next_cells = lines[next_index]
            if (
                len(next_cells) < 2
                or next_y0 - lines[next_index - 1][1] > table_line_gap
            ):
                break
            table_rows.append(next_cells)
            next_index += 1

        if len(table_rows) >= 2 or len(cells) >= 3:
            positioned.append((y0, lines[next_index - 1][1], _format_table(table_rows)))
            index = next_index
        else:
            positioned.append((y0, y1, " ".join(cells)))
            index += 1

    positioned.sort(key=lambda item: item[0])
    output: list[str] = []
    previous_y1: float | None = None
    for y0, y1, text in positioned:
        if previous_y1 is not None:
            separator = "\n\n" if y0 - previous_y1 > line_height * 1.5 else "\n"
            output.append(separator)
        output.append(text)
        previous_y1 = y1
    return "".join(output).strip()


def _pdf_date(value: str | None) -> date | None:
    if not value:
        return None
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if iso_match:
        try:
            return date(*(int(part) for part in iso_match.groups()))
        except ValueError:
            return None
    match = re.search(r"(?:D:)?(\d{4})(\d{2})?(\d{2})?", value)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month or "1"), int(day or "1"))
    except ValueError:
        return None


def metadata_from_pdf(
    file_path: Path, source_root: Path, document_metadata: dict[str, str | None]
) -> DocumentMetadata:
    title = (document_metadata.get("title") or "").strip() or file_path.stem
    creator = (document_metadata.get("author") or "").strip() or None
    relative_parent = file_path.parent.relative_to(source_root)
    category = relative_parent.parts[0] if relative_parent.parts else None

    return DocumentMetadata(
        title=title,
        institution=creator,
        publication_date=_pdf_date(
            document_metadata.get("publicationDate") or document_metadata.get("date")
        ),
        legal_validity=None,
        category=category,
        language=(document_metadata.get("language") or "").strip() or None,
        version=(document_metadata.get("version") or "").strip() or None,
    )


def extract_pages(
    file_path: Path,
    source_root: Path,
    ocr_page: OCRCallback,
    minimum_native_characters: int = 40,
) -> tuple[tuple[PageContent, ...], DocumentMetadata]:
    pages: list[PageContent] = []
    with fitz.open(file_path) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF contains no pages: {file_path}")
        metadata = metadata_from_pdf(
            file_path, source_root, document.metadata or {}
        )
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            native_text = _native_page_text(page)
            issues = native_text_quality_issues(
                native_text, minimum_characters=minimum_native_characters
            )
            text = native_text
            used_ocr = False
            if issues:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                text = ocr_page(pixmap.tobytes("png"), page_index + 1, file_path)
                used_ocr = True
            pages.append(
                PageContent(
                    page_number=page_index + 1,
                    text=text.strip(),
                    native_text=native_text,
                    used_ocr=used_ocr,
                )
            )
        page_count = document.page_count

    result = tuple(pages)
    validate_page_sequence(result, page_count)
    return result, metadata
