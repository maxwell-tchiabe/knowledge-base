from __future__ import annotations

import argparse
import logging

from knowledge_ingest.config import Settings, resolve_source_directory
from knowledge_ingest.pipeline import ingest_directory
from knowledge_ingest.storage import PostgresStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb-ingest",
        description="Extract local PDFs and index page-aware chunks in pgvector.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create the PostgreSQL/pgvector schema.")
    ingest = commands.add_parser("ingest", help="Ingest PDFs from a directory.")
    ingest.add_argument(
        "--source-dir",
        default="Dokumente",
        help="PDF root directory (default: Dokumente).",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and replace documents even when their hash is unchanged.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parser().parse_args()
    settings = Settings.from_environment()
    if args.command == "init-db":
        PostgresStore(settings.database_url).initialize()
        logging.info("Database schema is ready.")
        return

    source_root = resolve_source_directory(args.source_dir)
    summary = ingest_directory(source_root, settings, force=args.force)
    logging.info(
        "Ingestion complete: %d discovered, %d ingested, %d unchanged, %d chunks",
        summary.discovered,
        summary.ingested,
        summary.skipped_unchanged,
        summary.chunks_stored,
    )


if __name__ == "__main__":
    main()
