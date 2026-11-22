"""Safely remove one document from every project data store.

The default mode is a read-only preview. Pass --confirm to perform deletion.
The PDF, extraction cache, and metadata backup are archived under data/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.config import settings
from graph_rag.database.db import get_db_connection, init_db
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.ingestion.metadata_manager import metadata_manager


def _cache_path(filename: str) -> Path:
    key = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return settings.EXTRACTION_CACHE_DIR / f"{key}.json"


def _chunk_count(filename: str) -> int:
    rows = vector_store.collection.get(
        where={"document_name": filename},
        include=[],
    )
    return len(rows.get("ids") or [])


def _upload_count(filename: str) -> int:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM uploaded_documents WHERE filename = ?",
            (filename,),
        ).fetchone()
    return int(row["count"] if row else 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove one exact PDF from Neo4j, ChromaDB, metadata, SQLite, and cache"
    )
    parser.add_argument("filename", help="Exact PDF filename as stored in pdfs/metadata.csv")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Perform deletion; without this flag the command is read-only",
    )
    args = parser.parse_args()

    filename = Path(args.filename).name
    if filename != args.filename or not filename.lower().endswith(".pdf"):
        parser.error("Pass one PDF filename only, without a directory path")

    init_db()
    pdf_path = settings.PDF_DIR / filename
    cache_path = _cache_path(filename)
    metadata_record = metadata_manager.find_by_filename(filename)
    chunks = _chunk_count(filename)
    uploads = _upload_count(filename)

    print("Document removal preview")
    print(f"  Filename:          {filename}")
    print(f"  PDF present:       {pdf_path.exists()}")
    print(f"  Metadata rows:     {1 if metadata_record else 0}")
    print(f"  ChromaDB chunks:   {chunks}")
    print(f"  SQLite uploads:    {uploads}")
    print(f"  Extraction cache: {cache_path.exists()}")
    print(f"  Neo4j dataset:     {settings.DATASET_ID}")

    if not args.confirm:
        print("Preview only. Add --confirm to remove this document.")
        return 0

    if not graph_builder.delete_document(filename):
        print("ERROR: Neo4j deletion failed. No local data was removed.")
        return 1

    backup_root = settings.DATA_DIR / "deleted_documents"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"{stamp}_{Path(filename).stem}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    if settings.METADATA_FILE.exists():
        shutil.copy2(settings.METADATA_FILE, backup_dir / "metadata.csv.bak")

    vector_store.delete_document_chunks(filename)
    removed_metadata = metadata_manager.remove_document(filename)
    with get_db_connection() as conn:
        removed_uploads = conn.execute(
            "DELETE FROM uploaded_documents WHERE filename = ?",
            (filename,),
        ).rowcount

    if cache_path.exists():
        shutil.move(str(cache_path), str(backup_dir / cache_path.name))
    if pdf_path.exists():
        shutil.move(str(pdf_path), str(backup_dir / filename))

    print("Document removed successfully")
    print(f"  ChromaDB chunks removed: {chunks}")
    print(f"  Metadata rows removed:   {removed_metadata}")
    print(f"  SQLite rows removed:     {removed_uploads}")
    print(f"  Recovery backup:         {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
