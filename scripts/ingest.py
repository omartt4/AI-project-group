"""
Batch Ingestion CLI script.

Indexes the documents in the `pdfs/` folder into ChromaDB and Neo4j.
Enforces deduplication: documents already indexed in `pdfs/metadata.csv` are skipped.

Usage:
    uv run python scripts/ingest.py [--limit N] [--rebuild-graph]
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_rag.config import settings
from graph_rag.logging_config import configure_logging
from graph_rag.database.db import init_db
from graph_rag.ingestion.pipeline import pipeline
from graph_rag.ingestion.metadata_manager import metadata_manager
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.llm.cost_tracker import cost_tracker


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Graph RAG Ingestion CLI")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of supported documents to process")
    parser.add_argument(
        "--file",
        dest="only_filename",
        default=None,
        help="Process one exact supported document filename from the configured folder",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Rebuild Neo4j from a complete compatible extraction cache",
    )
    parser.add_argument(
        "--refresh-extraction",
        action="store_true",
        help="Ignore cache and request a fresh document-level extraction",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Graph RAG Ingestion Pipeline")
    print(f"Document Directory: {settings.PDF_DIR}")
    print(f"Metadata File:    {settings.METADATA_FILE}")
    print(f"ChromaDB Path:    {settings.CHROMA_PERSIST_DIR}")
    print(f"Chroma Collection:{settings.CHROMA_COLLECTION_NAME}")
    print(f"Dataset ID:       {settings.DATASET_ID}")
    print(f"Extraction Model: {settings.EXTRACTION_MODEL or '[NOT CONFIGURED]'}")
    print(f"Chat Model:       {settings.CHAT_MODEL or '[NOT CONFIGURED]'}")
    print(f"Evaluation Model: {settings.EVALUATION_MODEL or '[NOT CONFIGURED]'}")
    print(f"Embedding Model:  {settings.EMBEDDING_MODEL or '[NOT CONFIGURED]'}")
    print(f"Neo4j URI:        {settings.NEO4J_URI}")
    print("=" * 65)

    # Initialize DB schemas
    init_db()

    # Pre-check Neo4j status
    neo4j_online = graph_builder.is_connected()
    print(f"Neo4j Status:     {'[ONLINE]' if neo4j_online else '[OFFLINE - will store in ChromaDB only]'}")

    def progress_callback(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}", flush=True)

    print("\nStarting batch ingestion...", flush=True)
    usage_start_id = cost_tracker.last_id()
    results = pipeline.ingest_batch_folder(
        progress_callback=progress_callback,
        limit=args.limit,
        only_filename=args.only_filename,
        rebuild_graph=args.rebuild_graph,
        refresh_extraction=args.refresh_extraction,
    )

    completed = [r for r in results if r.status == "completed"]
    skipped = [r for r in results if r.status == "skipped_duplicate"]
    failed = [r for r in results if r.status == "failed"]

    print("\n" + "=" * 65)
    print("Ingestion Summary")
    print(f"  - Completed: {len(completed)}")
    print(f"  - Skipped (already indexed): {len(skipped)}")
    print(f"  - Failed:    {len(failed)}")
    print(f"  - Total chunks in ChromaDB: {vector_store.get_total_chunks()}")
    if neo4j_online:
        stats = graph_builder.get_graph_stats()
        print(
            f"  - Neo4j graph nodes: {stats.get('nodes', 0)}, "
            f"edges: {stats.get('edges', 0)}, "
            f"top-level topics: {stats.get('topics', 0)}, "
            f"subtopics: {stats.get('subtopics', 0)}"
        )
    run_usage = cost_tracker.summary(since_id=usage_start_id)
    print(
        f"  - This run calculated API cost: ${run_usage['cost_usd']:.8f} "
        f"({run_usage['requests']} successful, {run_usage['failed_requests']} "
        f"failed, {run_usage['total_tokens']} known tokens)"
    )
    if run_usage["unknown_cost_requests"]:
        print(
            "    WARNING: "
            f"{run_usage['unknown_cost_requests']} request(s) have unknown cost "
            "because model pricing or token usage is unavailable."
        )
    for row in run_usage["breakdown"]:
        cost_label = (
            f"${float(row['cost_usd']):.8f}"
            if row["cost_usd"] is not None
            else "unknown cost"
        )
        print(
            f"      {row['role']}: {cost_label} "
            f"({int(row['requests'])} successful, "
            f"{int(row['failed_requests'])} failed, "
            f"{int(row['unknown_cost_requests'])} unknown-cost)"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
