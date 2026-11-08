import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from knowledge_ingest.artifacts import ArtifactRun
from knowledge_ingest.models import Chunk, DocumentMetadata, PageContent


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        title="Test document",
        institution=None,
        publication_date=None,
        legal_validity=None,
        category="Tests",
        language="de",
        version=None,
    )


def test_document_artifacts_save_pages_ocr_image_chunks_and_vectors(
    tmp_path: Path,
) -> None:
    run = ArtifactRun(tmp_path)
    writer = run.new_document("Tests/example.pdf", "abc123")
    pages = (
        PageContent(1, "Native text", "Native text", False),
        PageContent(2, "OCR text", "", True),
    )
    chunks = (Chunk(0, "One chunk", (1, 2), "Section"),)
    embeddings = ((0.25, 0.75),)

    writer.save_ocr_image(2, b"png-image")
    writer.save_extraction(
        file_name="example.pdf",
        page_count=2,
        metadata=_metadata(),
        pages=pages,
    )
    writer.save_chunks(chunks=chunks, embedding_model="test-model")
    writer.save_embeddings(chunks=chunks, embeddings=embeddings)
    writer.publish()
    writer.cleanup()

    assert run.has_document_artifacts("Tests/example.pdf")
    document_folder = next((run.current / "documents").iterdir())
    assert (document_folder / "ocr_images" / "page-0002.png").read_bytes() == b"png-image"
    saved_pages = [
        json.loads(line)
        for line in (document_folder / "pages.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [page["used_ocr"] for page in saved_pages] == [False, True]
    saved_chunks = [
        json.loads(line)
        for line in (document_folder / "chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert saved_chunks[0]["page_numbers"] == [1, 2]
    with np.load(document_folder / "embeddings.npz") as archive:
        assert archive["vectors"].shape == (1, 2)
        assert archive["chunk_indices"].tolist() == [0]


def test_archive_contains_artifacts_and_sql_but_not_source_pdfs(
    tmp_path: Path,
) -> None:
    run = ArtifactRun(tmp_path)
    writer = run.new_document("Tests/source.pdf", "abc123")
    writer.save_extraction(
        file_name="source.pdf",
        page_count=1,
        metadata=_metadata(),
        pages=(PageContent(1, "Text", "Text", False),),
    )
    writer.save_chunks(
        chunks=(Chunk(0, "Text", (1,), None),),
        embedding_model="test-model",
    )
    writer.save_embeddings(
        chunks=(Chunk(0, "Text", (1,), None),),
        embeddings=((0.5,),),
    )
    writer.publish()
    (run.current / "database").mkdir()
    run.database_dump.write_text("-- SQL dump\n", encoding="utf-8")

    archive_path = run.create_archive()

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "database/knowledge_base.sql" in names
        assert any(name.endswith("chunks.jsonl") for name in names)
        assert not any(name.lower().endswith(".pdf") for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source_pdfs_included"] is False
    assert (run.root / "latest.zip").is_file()


def test_sql_dump_uses_compose_pg_dump_and_propagates_errors(
    tmp_path: Path,
) -> None:
    run = ArtifactRun(tmp_path)

    with patch(
        "knowledge_ingest.artifacts.subprocess.run",
        return_value=type(
            "Completed", (), {"returncode": 0, "stdout": b"-- dump", "stderr": b""}
        )(),
    ) as run_process:
        run.save_database_dump("postgresql://knowledge:localdev@127.0.0.1:5433/knowledge")

    assert run.database_dump.read_bytes() == b"-- dump"
    assert run_process.call_args.args[0][:5] == [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
    ]
    assert run_process.call_args.args[0][5:] == [
        "pg_dump",
        "-U",
        "knowledge",
        "-d",
        "knowledge",
        "--no-owner",
        "--no-privileges",
    ]

    with patch(
        "knowledge_ingest.artifacts.subprocess.run",
        return_value=type(
            "Completed", (), {"returncode": 1, "stdout": b"", "stderr": b"dump error"}
        )(),
    ):
        with pytest.raises(RuntimeError, match="dump error"):
            run.save_database_dump(
                "postgresql://knowledge:localdev@127.0.0.1:5433/knowledge"
            )
