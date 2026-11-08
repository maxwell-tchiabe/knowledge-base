from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from knowledge_ingest.artifacts import ArtifactRun, log_artifact_paths
from knowledge_ingest.chunking import build_chunks
from knowledge_ingest.config import Settings
from knowledge_ingest.pdf import extract_pages
from knowledge_ingest.providers import embed_chunks, make_groq_ocr_callback
from knowledge_ingest.storage import PostgresStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    discovered: int
    ingested: int
    skipped_unchanged: int
    chunks_stored: int


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_directory(
    source_root: Path,
    settings: Settings,
    *,
    force: bool = False,
) -> IngestionSummary:
    source_root = source_root.resolve()
    pdf_paths = tuple(
        sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        )
    )
    store = PostgresStore(settings.database_url)
    store.initialize()
    artifact_run = ArtifactRun(
        source_root,
        embedding_model=settings.embedding_model,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    ingested = 0
    skipped = 0
    chunk_count = 0

    for file_path in pdf_paths:
        source_path = file_path.relative_to(source_root).as_posix()
        digest = sha256_file(file_path)
        if not force and store.is_unchanged(
            source_path, digest, settings.embedding_model
        ) and artifact_run.has_document_artifacts(source_path):
            logger.info("Skipping unchanged document: %s", source_path)
            skipped += 1
            continue

        writer = artifact_run.new_document(source_path, digest)
        try:
            logger.info("Extracting document: %s", source_path)
            groq_ocr_page = make_groq_ocr_callback(settings)

            def save_ocr_page(
                image_png: bytes, page_number: int, ocr_file_path: Path
            ) -> str:
                writer.save_ocr_image(page_number, image_png)
                return groq_ocr_page(image_png, page_number, ocr_file_path)

            pages, metadata = extract_pages(
                file_path, source_root, save_ocr_page
            )
            chunks = build_chunks(
                pages,
                chunk_size_tokens=settings.chunk_size_tokens,
                chunk_overlap_tokens=settings.chunk_overlap_tokens,
            )
            embeddings = embed_chunks(chunks, settings)
            writer.save_extraction(
                file_name=file_path.name,
                page_count=len(pages),
                metadata=metadata,
                pages=pages,
            )
            writer.save_chunks(
                chunks=chunks,
                embedding_model=settings.embedding_model,
            )
            writer.save_embeddings(chunks=chunks, embeddings=embeddings)
            store.replace_document(
                source_path=source_path,
                sha256=digest,
                page_count=len(pages),
                embedding_model=settings.embedding_model,
                metadata=metadata,
                chunks=chunks,
                embeddings=embeddings,
            )
            writer.publish()
        finally:
            writer.cleanup()
        pages_with_ocr = sum(page.used_ocr for page in pages)
        logger.info(
            "Indexed %s: %d pages, %d OCR pages, %d chunks",
            source_path,
            len(pages),
            pages_with_ocr,
            len(chunks),
        )
        ingested += 1
        chunk_count += len(chunks)

    artifact_run.save_database_dump(settings.database_url)
    archive_path = artifact_run.create_archive()
    log_artifact_paths(artifact_run, archive_path)

    return IngestionSummary(
        discovered=len(pdf_paths),
        ingested=ingested,
        skipped_unchanged=skipped,
        chunks_stored=chunk_count,
    )
