# Knowledge base ingestion

This project indexes local PDF files into PostgreSQL with pgvector. It extracts text with PyMuPDF, calls Groq for OCR only on pages with sparse or suspicious native text, keeps page references on chunks, and generates embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`. PDF chunks are not sent to Groq for embedding; only pages that need OCR are sent to Groq.

The initial source tree is `Dokumente/`. PDFs are read only; ingestion does not modify or delete source files. URL/HTML collection, retrieval, and reranking are not included.

## Local setup

1. Install Python 3.11 or newer.
2. Copy `.env.example` to `.env`. Set `GROQ_API_KEY` and `GROQ_OCR_MODEL` if any pages need OCR. `EMBEDDING_MODEL` defaults to `sentence-transformers/all-MiniLM-L6-v2`. The first run downloads this model from Hugging Face; subsequent embedding runs use the cached local model. Do not commit `.env` or paste API keys into source files.
3. Start the local development database with `docker compose up -d postgres`. The compose configuration binds PostgreSQL to localhost and uses `localdev` as a development-only default password. For anything beyond local development, choose a strong password and secure the database separately.
4. Install the project and test dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   ```

5. Initialize the schema and ingest the PDFs:

   ```powershell
   kb-ingest init-db
   kb-ingest ingest
   ```

`DATABASE_URL` must match the database credentials. For the local defaults:

```text
postgresql://knowledge:localdev@localhost:5432/knowledge
```

Set `CHUNK_SIZE_TOKENS` and `CHUNK_OVERLAP_TOKENS` to tune chunking; defaults are 512 and 80. A financial table is kept together even when it exceeds the normal chunk size. Oversized prose is split as a fallback. Every stored chunk includes its page numbers and section title when available.

Run `kb-ingest ingest --source-dir <path>` to select a different PDF root. `--force` re-extracts documents even when their SHA-256 hash has not changed. Otherwise, a matching source path and content hash are skipped. Changed documents are embedded before a transaction replaces the stored document metadata and chunks, so provider failures leave the previous indexed version intact.

The `documents` table keeps a stable record per source-relative path, its current SHA-256 hash, and the embedding model used. `document_chunks` holds page-aware text and 384-dimensional vectors. Switching embedding models causes the document to be re-embedded on its next ingestion. On schema initialization, vectors from an incompatible embedding dimension are discarded and documents become eligible for re-indexing because vectors from different embedding spaces cannot be converted. Metadata fields that cannot be reliably read from the PDF remain null; source URL is reserved for future remote collection.

OCR needs a Groq model that accepts image input. Configure its exact model name in `GROQ_OCR_MODEL`; ingestion reports a clear error if OCR is required and the API key/model is missing or the provider call fails. The local embedding model runs on CPU. Embedding and database errors are surfaced and stop that document's ingestion.

## Saved ingestion artifacts and backups

Each successful `kb-ingest ingest` run saves generated files under `Dokumente/_ingestion/`:

- `current/documents/<document>/pages.jsonl` — extracted page text and whether OCR was used.
- `current/documents/<document>/ocr_images/` — PNGs only for pages sent through OCR.
- `current/documents/<document>/chunks.jsonl` — chunk text, page references, and section headings.
- `current/documents/<document>/embeddings.npz` — compressed 384-dimensional `float32` vectors and their chunk indexes.
- `current/documents/<document>/metadata.json` and `embedding_metadata.json` — source hash, document metadata, and model details.
- `current/database/knowledge_base.sql` — plain-text PostgreSQL dump of schema and indexed data.
- `archives/knowledge-base-<timestamp>.zip` and `latest.zip` — zipped generated artifacts and SQL dump. Original PDFs and `.env` are not copied into the archive.

The archive is created after a successful database dump. The first run after enabling artifact saving reprocesses documents whose generated artifacts do not exist yet; later unchanged runs reuse those artifacts and still refresh the SQL dump/archive. To restore the database, start the pgvector container, then run `psql` against the target database with `-f Dokumente/_ingestion/current/database/knowledge_base.sql`. Keep the `.sql` dump or zip outside the machine if you need it to survive local disk loss.

## Tests

```powershell
python -m pytest
```
