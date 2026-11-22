"""Rebuild only Topic/Subtopic hierarchy links from local metadata.

Usage:
    uv run python scripts/rebuild_topic_hierarchy.py

This migration does not parse PDFs, generate embeddings, or call an LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.graph.graph_builder import graph_builder
from graph_rag.ingestion.metadata_manager import metadata_manager


def main() -> int:
    records = metadata_manager.get_all_records()
    usable = [
        record
        for record in records
        if str(record.get("local_pdf") or "").strip()
        and str(record.get("title") or "").strip()
        and str(record.get("topic") or "").strip()
    ]
    if not usable:
        print("No metadata rows with filename, title, and topic were found.")
        return 1
    if not graph_builder.is_connected():
        print("Neo4j is unavailable. Check .env and run test_connections.py.")
        return 1

    graph_builder.init_constraints()
    for index, record in enumerate(usable, 1):
        filename = str(record["local_pdf"]).strip()
        graph_builder.add_document_node(
            document_name=filename,
            title=str(record["title"]).strip(),
            metadata={
                "topic": str(record["topic"]).strip(),
                "source": str(record.get("source") or "initial").strip(),
                "topic_source": "metadata_hierarchy_migration",
            },
        )
        print(f"[{index}/{len(usable)}] Rebuilt topic hierarchy: {filename}")

    classified_entities = graph_builder.classify_existing_topics()
    stats = graph_builder.get_graph_stats()
    print(
        "Finished: "
        f"{stats.get('topics', 0)} top-level topics, "
        f"{stats.get('subtopics', 0)} subtopics, "
        f"{classified_entities} existing topic entities classified"
    )
    graph_builder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
