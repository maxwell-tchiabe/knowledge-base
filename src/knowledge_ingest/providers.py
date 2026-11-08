from __future__ import annotations

import base64
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from groq import Groq
from sentence_transformers import SentenceTransformer

from knowledge_ingest.config import Settings
from knowledge_ingest.models import Chunk
from knowledge_ingest.pdf import OCRCallback

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


def make_groq_ocr_callback(settings: Settings) -> OCRCallback:
    client: Groq | None = None

    def recognize(image_png: bytes, page_number: int, file_path: Path) -> str:
        nonlocal client
        if not settings.groq_api_key:
            raise ValueError(
                f"Page {page_number} of {file_path} needs OCR, but GROQ_API_KEY "
                "is not configured."
            )
        if not settings.groq_ocr_model:
            raise ValueError(
                "Page-level OCR is needed, but GROQ_OCR_MODEL is not configured."
            )
        if client is None:
            client = Groq(api_key=settings.groq_api_key)

        encoded_image = base64.b64encode(image_png).decode("ascii")
        response = client.chat.completions.create(
            model=settings.groq_ocr_model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe all readable text on this document page. "
                                "Preserve headings, table rows, financial figures, "
                                "formulas, and their order. Do not summarize or infer "
                                "missing content."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError(
                f"Groq returned no OCR text for page {page_number} of {file_path}."
            )
        return text.strip()

    return recognize


def embed_chunks(
    chunks: Sequence[Chunk], settings: Settings
) -> tuple[tuple[float, ...], ...]:
    if not chunks:
        return ()
    if settings.embedding_model != EMBEDDING_MODEL:
        raise ValueError(
            f"Unsupported embedding model {settings.embedding_model!r}; "
            f"this schema is configured for {EMBEDDING_MODEL!r}."
        )

    model = _embedding_model(settings.embedding_model)
    encoded = model.encode(
        [chunk.content for chunk in chunks],
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if len(encoded) != len(chunks):
        raise RuntimeError(
            f"Embedding model returned {len(encoded)} vectors for {len(chunks)} chunks."
        )

    embeddings = tuple(
        tuple(float(value) for value in vector)
        for vector in encoded
    )
    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in embeddings):
        dimensions = sorted({len(vector) for vector in embeddings})
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS} embedding dimensions, "
            f"received {dimensions}."
        )
    return embeddings
