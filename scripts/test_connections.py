"""Verify OpenRouter and Neo4j configuration without ingesting documents.

Usage:
    uv run python scripts/test_connections.py
    uv run python scripts/test_connections.py --deep

The OpenRouter checks only list available models. They do not run inference or
create embeddings unless ``--deep`` is supplied. The Neo4j check runs a
read-only ``RETURN 1`` query.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Set

import httpx
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.config import settings
from graph_rag.database.db import init_db
from graph_rag.llm.model_client import (
    chat_model_client,
    embedding_model_client,
    evaluation_model_client,
    extraction_model_client,
)
from graph_rag.llm.cost_tracker import cost_tracker


def _model_ids(path: str) -> Set[str]:
    """Return model IDs exposed by one OpenRouter discovery endpoint."""
    headers = {
        "Authorization": f"Bearer {settings.MODEL_API_KEY}",
        "X-Title": settings.OPENROUTER_APP_NAME,
    }
    if settings.OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL

    with httpx.Client(
        base_url=settings.MODEL_API_BASE_URL.rstrip("/"),
        headers=headers,
        timeout=20.0,
    ) as client:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return {
        str(row.get("id"))
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def _report_models(
    available: Set[str],
    configured: Iterable[tuple[str, str]],
) -> bool:
    ok = True
    for role, model_id in configured:
        if not model_id:
            print(f"[FAIL] {role}: model ID is missing from .env")
            ok = False
        elif model_id not in available:
            print(f"[FAIL] {role}: configured model is not available: {model_id}")
            ok = False
        else:
            print(f"[OK]   {role}: {model_id}")
    return ok


def check_openrouter() -> bool:
    """Check API authentication and availability of all configured models."""
    print("\nOpenRouter")
    print("----------")
    if not settings.MODEL_API_KEY:
        print("[FAIL] OPENROUTER_API_KEY is missing from .env")
        return False

    try:
        chat_models = _model_ids("/models")
        embedding_models = _model_ids("/embeddings/models")
    except (httpx.HTTPError, ValueError) as exc:
        print(f"[FAIL] Could not authenticate or list models: {exc}")
        return False

    chat_ok = _report_models(
        chat_models,
        (
            ("extraction", settings.EXTRACTION_MODEL),
            ("chat", settings.CHAT_MODEL),
            ("evaluation", settings.EVALUATION_MODEL),
        ),
    )
    embedding_ok = _report_models(
        embedding_models,
        (("embedding", settings.EMBEDDING_MODEL),),
    )
    return chat_ok and embedding_ok


def check_inference() -> bool:
    """Run one minimal paid request per unique configured model."""
    print("\nOpenRouter deep inference check")
    print("-------------------------------")
    unique_chat_models = {}
    for role, client in (
        ("extraction", extraction_model_client),
        ("chat", chat_model_client),
        ("evaluation", evaluation_model_client),
    ):
        unique_chat_models.setdefault(client.llm_model, (role, client))
    try:
        for model, (role, client) in unique_chat_models.items():
            client.generate("Reply with OK only.", temperature=0.0)
            print(f"[OK]   inference: {model} ({role})")
        embedding_model_client.get_embedding("connection test")
        print(f"[OK]   embedding inference: {settings.EMBEDDING_MODEL}")
        return True
    except Exception as exc:
        print(f"[FAIL] Paid inference request failed: {exc}")
        return False


def check_neo4j() -> bool:
    """Verify Neo4j credentials with a read-only query."""
    print("\nNeo4j")
    print("-----")
    if not all((settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)):
        print("[FAIL] Neo4j URI, username, or password is missing from .env")
        return False

    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=10.0,
        )
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE or None) as session:
            value = session.run("RETURN 1 AS connection_test").single()[
                "connection_test"
            ]
        if value != 1:
            raise RuntimeError("Neo4j returned an unexpected test value")
        print("[OK]   Credentials accepted and the database is reachable")
        return True
    except Exception as exc:
        print(f"[FAIL] Connection failed: {exc}")
        return False
    finally:
        if driver is not None:
            driver.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Make minimal paid inference calls to verify credits and permissions",
    )
    args = parser.parse_args()
    init_db()
    print("Connection checks only - no PDFs will be processed.")
    openrouter_ok = check_openrouter()
    if openrouter_ok and args.deep:
        usage_start_id = cost_tracker.last_id()
        openrouter_ok = check_inference()
        usage = cost_tracker.summary(since_id=usage_start_id)
        print(
            f"Deep-check calculated cost: ${usage['cost_usd']:.8f} "
            f"({usage['requests']} successful, {usage['failed_requests']} failed, "
            f"{usage['total_tokens']} known tokens)"
        )
        if usage["unknown_cost_requests"]:
            print(
                f"WARNING: {usage['unknown_cost_requests']} request(s) have unknown "
                "cost because model pricing or token usage is unavailable."
            )
    neo4j_ok = check_neo4j()

    print("\nResult")
    print("------")
    if openrouter_ok and neo4j_ok:
        print("All configured services are ready.")
        return 0

    print("One or more checks failed. Fix the reported setting before ingestion.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
