"""Safely remove one document from every project data store.

The default mode is a read-only preview. Pass --confirm to perform deletion.
The PDF, extraction cache, and metadata backup are archived under data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.config import settings
from graph_rag.database.db import (
    delete_document_pages,
    delete_document_evidence,
    delete_structured_tables,
    delete_unresolved_references,
    get_db_connection,
    init_db,
)
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.ingestion.file_handler import file_handler
from graph_rag.ingestion.metadata_manager import metadata_manager


def _cache_path(document_id: str) -> Path:
    key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
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
        description="Remove one exact supported document from Neo4j, ChromaDB, metadata, SQLite, and cache"
    )
    parser.add_argument("filename", help="Exact supported document filename as stored in metadata")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Perform deletion; without this flag the command is read-only",
    )
    args = parser.parse_args()

    filename = Path(args.filename).name
    if filename != args.filename or not file_handler.is_supported(filename):
        parser.error("Pass one supported document filename only, without a directory path")

    init_db()
    document_path = settings.PDF_DIR / filename
    metadata_record = metadata_manager.find_by_filename(filename)
    document_id = ""
    if document_path.exists():
        hasher = hashlib.sha256()
        with document_path.open("rb") as document_file:
            for block in iter(lambda: document_file.read(65536), b""):
                hasher.update(block)
        document_id = hasher.hexdigest()
    else:
        document_id = str((metadata_record or {}).get("file_hash") or "")
    cache_path = _cache_path(document_id) if document_id else None
    chunks = _chunk_count(filename)
    uploads = _upload_count(filename)

    print("Document removal preview")
    print(f"  Filename:          {filename}")
    print(f"  File present:      {document_path.exists()}")
    print(f"  Metadata rows:     {1 if metadata_record else 0}")
    print(f"  ChromaDB chunks:   {chunks}")
    print(f"  SQLite uploads:    {uploads}")
    print(f"  Extraction cache: {bool(cache_path and cache_path.exists())}")
    print(f"  Neo4j dataset:     {settings.DATASET_ID}")

    if not args.confirm:
        print("Preview only. Add --confirm to remove this document.")
        return 0

    if not graph_builder.delete_document(filename, document_id=document_id):
        print("ERROR: Neo4j deletion failed. No local data was removed.")
        return 1

    backup_root = settings.DATA_DIR / "deleted_documents"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"{stamp}_{Path(filename).stem}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    if settings.METADATA_FILE.exists():
        shutil.copy2(settings.METADATA_FILE, backup_dir / "metadata.csv.bak")

    with get_db_connection() as conn:
        structured_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT document_name, file_hash, table_name, columns_json, rows_json "
                "FROM structured_tables WHERE document_name = ?",
                (filename,),
            ).fetchall()
        ]
        page_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT document_name, file_hash, page_number, sheet_name, section, "
                "text, ocr_text, source_offsets_json, bounding_boxes_json, "
                "table_data_json, extraction_method, confidence, language "
                "FROM document_pages WHERE document_name = ?",
                (filename,),
            ).fetchall()
        ]
        unresolved_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT document_name, file_hash, mention_id, mention_text, "
                "entity_name, entity_category, mention_type, evidence, confidence, "
                "attributes_json, resolution_status, created_at "
                "FROM unresolved_references WHERE document_name = ?",
                (filename,),
            ).fetchall()
        ]
        evidence_rows = (
            [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT document_id, document_name, file_hash, object_type,
                           object_id, mention_text, evidence_text, page_number,
                           sheet_name, section, source_offsets_json,
                           bounding_boxes_json, confidence, extraction_model,
                           extraction_version, created_at
                    FROM graph_evidence
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchall()
            ]
            if document_id
            else []
        )
        diagnostic_rows = (
            [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT document_id, document_name, file_hash,
                           diagnostic_type, details_json, evidence, confidence,
                           created_at
                    FROM extraction_diagnostics
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchall()
            ]
            if document_id
            else []
        )
    (backup_dir / "structured_tables.json").write_text(
        json.dumps(structured_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (backup_dir / "document_pages.json").write_text(
        json.dumps(page_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (backup_dir / "unresolved_references.json").write_text(
        json.dumps(unresolved_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (backup_dir / "graph_evidence.json").write_text(
        json.dumps(evidence_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (backup_dir / "extraction_diagnostics.json").write_text(
        json.dumps(diagnostic_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vector_store.delete_document_chunks(filename)
    if document_id:
        vector_store.delete_document_entity_vectors(document_id)
    removed_metadata = metadata_manager.remove_document(filename)
    with get_db_connection() as conn:
        removed_uploads = conn.execute(
            "DELETE FROM uploaded_documents WHERE filename = ?",
            (filename,),
        ).rowcount
    delete_structured_tables(filename)
    delete_document_pages(filename)
    delete_unresolved_references(filename)
    if document_id:
        delete_document_evidence(document_id)

    if cache_path and cache_path.exists():
        shutil.move(str(cache_path), str(backup_dir / cache_path.name))
    if document_path.exists():
        shutil.move(str(document_path), str(backup_dir / filename))

    print("Document removed successfully")
    print(f"  ChromaDB chunks removed: {chunks}")
    print(f"  Metadata rows removed:   {removed_metadata}")
    print(f"  SQLite rows removed:     {removed_uploads}")
    print(f"  Recovery backup:         {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
