from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class PageContent:
    page_number: int
    text: str
    native_text: str
    used_ocr: bool


@dataclass(frozen=True)
class ContentUnit:
    text: str
    page_number: int
    section_title: str | None
    is_table: bool = False


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    content: str
    page_numbers: tuple[int, ...]
    section_title: str | None


@dataclass(frozen=True)
class DocumentMetadata:
    title: str
    institution: str | None
    publication_date: date | None
    legal_validity: str | None
    category: str | None
    language: str | None
    version: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "institution": self.institution,
            "publication_date": (
                self.publication_date.isoformat()
                if self.publication_date is not None
                else None
            ),
            "legal_validity": self.legal_validity,
            "category": self.category,
            "language": self.language,
            "version": self.version,
        }


@dataclass(frozen=True)
class ExtractedDocument:
    source_path: str
    file_path: Path
    sha256: str
    page_count: int
    metadata: DocumentMetadata
    pages: tuple[PageContent, ...]
    chunks: tuple[Chunk, ...]
