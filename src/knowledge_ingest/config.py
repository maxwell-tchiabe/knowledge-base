from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    groq_api_key: str | None
    groq_ocr_model: str | None
    embedding_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    @classmethod
    def from_environment(cls) -> Settings:
        load_dotenv()
        chunk_size = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
        if chunk_size < 1:
            raise ValueError("CHUNK_SIZE_TOKENS must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be at least zero and smaller than "
                "CHUNK_SIZE_TOKENS."
            )

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required.")

        return cls(
            database_url=database_url,
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_ocr_model=os.getenv("GROQ_OCR_MODEL") or None,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
        )


def resolve_source_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"PDF source directory does not exist: {path}")
    return path
