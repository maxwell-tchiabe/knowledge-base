from unittest.mock import patch

import pytest

from knowledge_ingest.config import Settings
from knowledge_ingest.models import Chunk
from knowledge_ingest.providers import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_chunks,
)


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://unused",
        groq_api_key=None,
        groq_ocr_model=None,
        embedding_model=EMBEDDING_MODEL,
        chunk_size_tokens=512,
        chunk_overlap_tokens=80,
    )


def _chunk(index: int) -> Chunk:
    return Chunk(index, f"Chunk {index}", (index + 1,), None)


def test_embeddings_use_local_model_and_keep_input_order() -> None:
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            assert texts == ["Chunk 0", "Chunk 1", "Chunk 2"]
            assert kwargs["normalize_embeddings"] is True
            return [
                [float(index)] * EMBEDDING_DIMENSIONS
                for index, _text in enumerate(texts)
            ]

    with patch("knowledge_ingest.providers._embedding_model", return_value=FakeModel()) as loader:
        vectors = embed_chunks(tuple(_chunk(index) for index in range(3)), _settings())

    loader.assert_called_once_with(EMBEDDING_MODEL)
    assert len(vectors) == 3
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)


def test_empty_embedding_input_does_not_load_model() -> None:
    with patch("knowledge_ingest.providers._embedding_model") as model_loader:
        assert embed_chunks((), _settings()) == ()
    model_loader.assert_not_called()


def test_embedding_dimension_mismatch_is_reported() -> None:
    class FakeModel:
        def encode(self, _texts: list[str], **_kwargs: object) -> list[list[float]]:
            return [[0.0] * 8]

    with patch("knowledge_ingest.providers._embedding_model", return_value=FakeModel()):
        with pytest.raises(RuntimeError, match="Expected 384 embedding dimensions"):
            embed_chunks((_chunk(0),), _settings())


def test_unsupported_local_embedding_model_is_rejected() -> None:
    settings = _settings()
    unsupported = Settings(
        database_url=settings.database_url,
        groq_api_key=settings.groq_api_key,
        groq_ocr_model=settings.groq_ocr_model,
        embedding_model="another/model",
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    with pytest.raises(ValueError, match="Unsupported embedding model"):
        embed_chunks((_chunk(0),), unsupported)
