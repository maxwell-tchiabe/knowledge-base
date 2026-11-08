from __future__ import annotations

from collections import Counter
from math import ceil

from knowledge_ingest.models import PageContent


def native_text_quality_issues(text: str, minimum_characters: int = 40) -> tuple[str, ...]:
    stripped = text.strip()
    if len(stripped) < minimum_characters:
        return ("sparse_or_missing_text",)

    suspicious_count = sum(
        character == "\ufffd"
        or (ord(character) < 32 and character not in "\n\r\t\f")
        for character in stripped
    )
    if suspicious_count / len(stripped) > 0.02:
        return ("suspicious_characters",)
    return ()


def validate_page_sequence(
    pages: tuple[PageContent, ...], expected_page_count: int
) -> None:
    actual = tuple(page.page_number for page in pages)
    expected = tuple(range(1, expected_page_count + 1))
    if actual != expected:
        raise ValueError(
            f"Page extraction mismatch: expected pages {expected}, found {actual}."
        )


def remove_repeated_page_edges(pages: tuple[PageContent, ...]) -> tuple[str, ...]:
    if len(pages) < 2:
        return tuple(_remove_consecutive_duplicates(page.text) for page in pages)

    edge_counts: Counter[str] = Counter()
    for page in pages:
        nonempty_lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if nonempty_lines:
            edge_counts.update({nonempty_lines[0], nonempty_lines[-1]})
    minimum_occurrences = max(2, ceil(len(pages) / 2))
    repeated = {
        line
        for line, count in edge_counts.items()
        if line and count >= minimum_occurrences
    }

    cleaned: list[str] = []
    for page in pages:
        lines = page.text.splitlines()
        while lines and lines[0].strip() in repeated:
            lines.pop(0)
        while lines and lines[-1].strip() in repeated:
            lines.pop()
        cleaned.append(_remove_consecutive_duplicates("\n".join(lines).strip()))
    return tuple(cleaned)


def _remove_consecutive_duplicates(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    previous: str | None = None
    for line in lines:
        normalized = line.strip()
        if normalized and normalized == previous and not normalized.startswith("|"):
            continue
        result.append(line)
        previous = normalized if normalized else None
    return "\n".join(result)
