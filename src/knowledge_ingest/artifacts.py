from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from knowledge_ingest.models import Chunk, DocumentMetadata, PageContent
from knowledge_ingest.providers import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class DocumentArtifactWriter:
    def __init__(
        self,
        run: ArtifactRun,
        source_path: str,
        sha256: str,
    ) -> None:
        self._run = run
        self.source_path = source_path
        self.sha256 = sha256
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
        source_name = Path(source_path).stem
        self.target = (
            run.current
            / "documents"
            / f"{source_name}-{source_hash}"
        )
        self._staging = Path(
            tempfile.mkdtemp(prefix=f"{source_hash}-", dir=run.staging)
        )
        self._published = False

    def save_ocr_image(self, page_number: int, image_png: bytes) -> None:
        images = self._staging / "ocr_images"
        images.mkdir(parents=True, exist_ok=True)
        (images / f"page-{page_number:04d}.png").write_bytes(image_png)

    def save_extraction(
        self,
        *,
        file_name: str,
        page_count: int,
        metadata: DocumentMetadata,
        pages: tuple[PageContent, ...],
    ) -> None:
        _write_json(
            self._staging / "metadata.json",
            {
                "source_path": self.source_path,
                "source_file_name": file_name,
                "sha256": self.sha256,
                "page_count": page_count,
                "metadata": metadata.as_dict(),
            },
        )
        pages_path = self._staging / "pages.jsonl"
        with pages_path.open("w", encoding="utf-8", newline="\n") as output:
            for page in pages:
                output.write(
                    json.dumps(
                        {
                            "page_number": page.page_number,
                            "text": page.text,
                            "native_text": page.native_text,
                            "used_ocr": page.used_ocr,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def save_chunks(
        self,
        *,
        chunks: tuple[Chunk, ...],
        embedding_model: str,
    ) -> None:
        chunks_path = self._staging / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                output.write(
                    json.dumps(
                        {
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "page_numbers": list(chunk.page_numbers),
                            "section_title": chunk.section_title,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _write_json(
            self._staging / "embedding_metadata.json",
            {
                "model": embedding_model,
                "dimensions": EMBEDDING_DIMENSIONS,
                "normalization": "l2",
                "format": "NumPy compressed archive (.npz), float32",
            },
        )

    def save_embeddings(
        self,
        *,
        chunks: tuple[Chunk, ...],
        embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} vectors."
            )
        np.savez_compressed(
            self._staging / "embeddings.npz",
            vectors=np.asarray(embeddings, dtype=np.float32),
            chunk_indices=np.asarray(
                [chunk.chunk_index for chunk in chunks], dtype=np.int32
            ),
        )

    def publish(self) -> None:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        backup = self.target.with_name(f"{self.target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if self.target.exists():
            os.replace(self.target, backup)
        try:
            os.replace(self._staging, self.target)
        except OSError:
            if backup.exists():
                os.replace(backup, self.target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        self._published = True

    def cleanup(self) -> None:
        if not self._published and self._staging.exists():
            shutil.rmtree(self._staging)


class ArtifactRun:
    def __init__(
        self,
        source_root: Path,
        *,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 80,
    ) -> None:
        self.root = source_root / "_ingestion"
        self.current = self.root / "current"
        self.staging = self.root / ".staging"
        self.root.mkdir(parents=True, exist_ok=True)
        self.current.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.embedding_model = embedding_model
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.database_dump = self.current / "database" / "knowledge_base.sql"

    def has_document_artifacts(self, source_path: str) -> bool:
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
        source_name = Path(source_path).stem
        folder = self.current / "documents" / f"{source_name}-{source_hash}"
        return all(
            (folder / name).is_file()
            for name in (
                "metadata.json",
                "pages.jsonl",
                "chunks.jsonl",
                "embedding_metadata.json",
                "embeddings.npz",
            )
        )

    def new_document(
        self,
        source_path: str,
        sha256: str,
    ) -> DocumentArtifactWriter:
        return DocumentArtifactWriter(self, source_path, sha256)

    def save_database_dump(self, database_url: str) -> None:
        parsed = urlparse(database_url)
        database_name = parsed.path.lstrip("/")
        if not database_name or "/" in database_name:
            raise ValueError("DATABASE_URL must specify one database name for pg_dump.")

        self.database_dump.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "pg_dump",
            "-U",
            parsed.username or "knowledge",
            "-d",
            database_name,
            "--no-owner",
            "--no-privileges",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=self.root.parent.parent,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"PostgreSQL SQL dump failed: {error}")
        if not result.stdout:
            raise RuntimeError("pg_dump succeeded but produced an empty SQL dump.")
        self.database_dump.write_bytes(result.stdout)

    def create_archive(self) -> Path:
        _write_json(
            self.current / "manifest.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "database_dump": "database/knowledge_base.sql",
                "document_artifacts": "documents/",
                "embedding_model": self.embedding_model,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "chunk_size_tokens": self.chunk_size_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
                "ocr_images": "Only pages routed through OCR are included.",
                "source_pdfs_included": False,
            },
        )
        archives = self.root / "archives"
        archives.mkdir(parents=True, exist_ok=True)
        archive_path = archives / f"knowledge-base-{self.run_id}.zip"
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for file_path in sorted(self.current.rglob("*")):
                if file_path.is_file():
                    archive.write(
                        file_path,
                        arcname=file_path.relative_to(self.current).as_posix(),
                    )

        latest_temp = self.root / "latest.zip.tmp"
        shutil.copyfile(archive_path, latest_temp)
        os.replace(latest_temp, self.root / "latest.zip")
        return archive_path


def log_artifact_paths(run: ArtifactRun, archive_path: Path) -> None:
    logger.info("Artifact directory: %s", run.current)
    logger.info("SQL dump: %s", run.database_dump)
    logger.info("Artifact archive: %s", archive_path)
