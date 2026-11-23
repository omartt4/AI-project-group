"""
Database module for SQLite storage.

Manages connections, schema initialization, and transactional access for:
- Users & Authentication
- Chat Sessions & Messages
- Live Session Evaluation Logs
- Document Upload Registry
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List
from graph_rag.config import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ── Users Table ──
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL COLLATE NOCASE,
    email           TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Chat Sessions Table ──
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Chat Messages Table ──
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    sources         TEXT, -- JSON string of retrieved chunks/entities
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── RAG Evaluation Logs Table ──
CREATE TABLE IF NOT EXISTS evaluation_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id          INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    session_id          INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    context_chunks      TEXT,    -- JSON array of retrieved context chunks
    faithfulness        REAL,    -- 0.0 to 1.0 (faithfulness to context)
    answer_relevance    REAL,    -- 0.0 to 1.0 (relevance to user query)
    context_relevance   REAL,    -- 0.0 to 1.0 (relevance of context to query)
    retrieval_ms        INTEGER, -- Retrieval latency in ms
    generation_ms       INTEGER, -- LLM generation latency in ms
    eval_ms             INTEGER, -- Evaluation latency in ms
    total_ms            INTEGER, -- Total turnaround latency in ms
    eval_reasoning      TEXT,    -- JSON string with judge commentary
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Uploaded Documents Table ──
CREATE TABLE IF NOT EXISTS uploaded_documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    filename            TEXT NOT NULL,
    file_type           TEXT NOT NULL,
    file_size_bytes     INTEGER,
    file_hash           TEXT NOT NULL,
    source_archive      TEXT,
    num_chunks          INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'pending', -- pending, processing, completed, failed, duplicate_skipped
    error_message       TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Per-request OpenRouter Usage & Cost ──
CREATE TABLE IF NOT EXISTS model_usage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    role                TEXT NOT NULL,
    operation           TEXT NOT NULL,
    model               TEXT NOT NULL,
    endpoint            TEXT NOT NULL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER,
    cost_usd            REAL,
    provider_cost_usd   REAL,
    input_rate_usd_per_million REAL,
    output_rate_usd_per_million REAL,
    cost_method         TEXT,
    pricing_version     TEXT,
    request_id          TEXT,
    run_id              TEXT,
    user_id             INTEGER,
    latency_ms          INTEGER,
    status              TEXT NOT NULL DEFAULT 'success',
    retry_count         INTEGER,
    cache_status        TEXT NOT NULL DEFAULT 'miss',
    error_message       TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Indexes for Fast Lookups ──
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_session ON evaluation_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_hash ON uploaded_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_model_usage_created ON model_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_model_usage_role ON model_usage(role);

-- ── Original structured rows from CSV, Excel, JSON, and Parquet ──
CREATE TABLE IF NOT EXISTS structured_tables (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name   TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    table_name      TEXT NOT NULL,
    columns_json    TEXT NOT NULL,
    rows_json       TEXT NOT NULL,
    UNIQUE(document_name, table_name)
);
CREATE INDEX IF NOT EXISTS idx_structured_tables_hash
    ON structured_tables(file_hash);

-- ── Page-level text and extraction provenance ──
CREATE TABLE IF NOT EXISTS document_pages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name       TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    page_number         INTEGER NOT NULL,
    sheet_name          TEXT NOT NULL DEFAULT '',
    section             TEXT NOT NULL DEFAULT '',
    text                TEXT NOT NULL,
    ocr_text            TEXT NOT NULL DEFAULT '',
    source_offsets_json TEXT NOT NULL DEFAULT '[]',
    bounding_boxes_json TEXT NOT NULL DEFAULT '[]',
    table_data_json     TEXT NOT NULL DEFAULT '[]',
    extraction_method   TEXT NOT NULL DEFAULT '',
    confidence          REAL,
    language            TEXT NOT NULL DEFAULT 'und',
    UNIQUE(document_name, page_number)
);
CREATE INDEX IF NOT EXISTS idx_document_pages_hash
    ON document_pages(file_hash);

-- ── Mentions that could not be resolved without guessing ──
CREATE TABLE IF NOT EXISTS unresolved_references (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name       TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    mention_id          TEXT NOT NULL,
    mention_text        TEXT NOT NULL,
    entity_name         TEXT NOT NULL DEFAULT '',
    entity_category     TEXT NOT NULL DEFAULT '',
    mention_type        TEXT NOT NULL DEFAULT '',
    evidence            TEXT NOT NULL,
    confidence          REAL NOT NULL DEFAULT 0.0,
    attributes_json     TEXT NOT NULL DEFAULT '{}',
    resolution_status   TEXT NOT NULL DEFAULT 'unresolved',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_name, mention_id)
);
CREATE INDEX IF NOT EXISTS idx_unresolved_references_hash
    ON unresolved_references(file_hash);

-- ── Durable cross-store ingestion progress ──
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    mode                TEXT NOT NULL,
    document_name       TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    status              TEXT NOT NULL,
    stage               TEXT NOT NULL,
    attempt_count       INTEGER NOT NULL DEFAULT 1,
    error_message       TEXT NOT NULL DEFAULT '',
    started_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_document_hash
    ON ingestion_runs(document_name, file_hash, id);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status
    ON ingestion_runs(status, updated_at);

-- ── Evidence and non-blocking extraction diagnostics ──
CREATE TABLE IF NOT EXISTS graph_evidence (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         TEXT NOT NULL,
    document_name       TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    object_type         TEXT NOT NULL,
    object_id           TEXT NOT NULL,
    mention_text        TEXT NOT NULL DEFAULT '',
    evidence_text       TEXT NOT NULL,
    page_number         INTEGER,
    sheet_name          TEXT NOT NULL DEFAULT '',
    section             TEXT NOT NULL DEFAULT '',
    source_offsets_json TEXT NOT NULL DEFAULT '[]',
    bounding_boxes_json TEXT NOT NULL DEFAULT '[]',
    confidence          REAL NOT NULL,
    extraction_model    TEXT NOT NULL,
    extraction_version  TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_graph_evidence_document
    ON graph_evidence(document_id, object_type);

CREATE TABLE IF NOT EXISTS extraction_diagnostics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL,
    document_name   TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    diagnostic_type TEXT NOT NULL,
    details_json    TEXT NOT NULL,
    evidence        TEXT NOT NULL DEFAULT '',
    confidence      REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_extraction_diagnostics_document
    ON extraction_diagnostics(document_id, diagnostic_type);

CREATE TABLE IF NOT EXISTS extraction_locks (
    document_hash   TEXT PRIMARY KEY,
    owner_token     TEXT NOT NULL,
    expires_at      REAL NOT NULL
);
"""


def get_raw_connection() -> sqlite3.Connection:
    """Return a raw SQLite connection with row factory configured."""
    settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        settings.SQLITE_DB_PATH,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for transactional database operations."""
    conn = get_raw_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database schemas and tables if they do not exist."""
    settings.ensure_directories()
    with get_db_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        evidence_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(graph_evidence)").fetchall()
        }
        if "mention_text" not in evidence_columns:
            conn.execute(
                "ALTER TABLE graph_evidence ADD COLUMN mention_text TEXT NOT NULL DEFAULT ''"
            )
        usage_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(model_usage)").fetchall()
        }
        migrations = {
            "request_id": "TEXT",
            "run_id": "TEXT",
            "user_id": "INTEGER",
            "latency_ms": "INTEGER",
            "status": "TEXT NOT NULL DEFAULT 'success'",
            "retry_count": "INTEGER",
            "cache_status": "TEXT NOT NULL DEFAULT 'miss'",
            "error_message": "TEXT",
            "provider_cost_usd": "REAL",
            "input_rate_usd_per_million": "REAL",
            "output_rate_usd_per_million": "REAL",
            "cost_method": "TEXT",
            "pricing_version": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in usage_columns:
                conn.execute(
                    f"ALTER TABLE model_usage ADD COLUMN {column} {definition}"
                )
        from graph_rag.llm.pricing import PRICING_VERSION, calculate_cost_usd

        legacy_rows = conn.execute(
            """
            SELECT id, model, prompt_tokens, completion_tokens, cost_usd
            FROM model_usage
            WHERE cost_method IS NULL
            """
        ).fetchall()
        for row in legacy_rows:
            calculation = calculate_cost_usd(
                row["model"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
            )
            conn.execute(
                """
                UPDATE model_usage
                SET provider_cost_usd = COALESCE(provider_cost_usd, cost_usd),
                    cost_usd = ?,
                    input_rate_usd_per_million = ?,
                    output_rate_usd_per_million = ?,
                    cost_method = ?,
                    pricing_version = ?
                WHERE id = ?
                """,
                (
                    (
                        float(calculation.cost_usd)
                        if calculation.cost_usd is not None
                        else None
                    ),
                    (
                        float(calculation.input_usd_per_million)
                        if calculation.input_usd_per_million is not None
                        else None
                    ),
                    (
                        float(calculation.output_usd_per_million)
                        if calculation.output_usd_per_million is not None
                        else None
                    ),
                    calculation.method,
                    (
                        PRICING_VERSION
                        if calculation.input_usd_per_million is not None
                        else None
                    ),
                    row["id"],
                ),
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_usage_request ON model_usage(request_id)"
        )


def save_structured_tables(
    document_name: str,
    file_hash: str,
    tables: List[Dict[str, Any]],
) -> None:
    """Replace the stored structured rows for one document atomically."""
    rows = [
        (
            document_name,
            file_hash,
            str(table["table_name"]),
            json.dumps(table["columns"], ensure_ascii=False, default=str),
            json.dumps(table["rows"], ensure_ascii=False, default=str),
        )
        for table in tables
    ]
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM structured_tables WHERE document_name = ?",
            (document_name,),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO structured_tables
                (document_name, file_hash, table_name, columns_json, rows_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )


def delete_structured_tables(document_name: str) -> None:
    """Delete stored structured rows for one exact document."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM structured_tables WHERE document_name = ?",
            (document_name,),
        )


def save_document_pages(
    document_name: str,
    file_hash: str,
    pages: List[Any],
) -> None:
    """Replace page text and provenance for one document atomically."""
    rows = [
        (
            document_name,
            file_hash,
            int(page.page_number),
            str(page.sheet_name or ""),
            str(page.section or ""),
            str(page.text or ""),
            str(page.ocr_text or ""),
            json.dumps(page.source_offsets, ensure_ascii=False, default=str),
            json.dumps(page.bounding_boxes, ensure_ascii=False, default=str),
            json.dumps(page.table_data, ensure_ascii=False, default=str),
            str(page.extraction_method or ""),
            page.confidence,
            str(page.language or "und"),
        )
        for page in pages
    ]
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM document_pages WHERE document_name = ?",
            (document_name,),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO document_pages
                (document_name, file_hash, page_number, sheet_name, section,
                 text, ocr_text, source_offsets_json, bounding_boxes_json,
                 table_data_json, extraction_method, confidence, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def delete_document_pages(document_name: str) -> None:
    """Delete page-level extraction data for one exact document."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM document_pages WHERE document_name = ?",
            (document_name,),
        )


def save_unresolved_references(
    document_name: str,
    file_hash: str,
    mentions: List[Any],
) -> None:
    """Replace unresolved mention evidence for one document atomically."""
    rows = [
        (
            document_name,
            file_hash,
            str(mention.mention_id),
            str(mention.text),
            str(mention.entity_name or ""),
            str(mention.entity_category or ""),
            str(mention.mention_type or ""),
            str(mention.evidence),
            float(mention.confidence or 0.0),
            json.dumps(mention.attributes or {}, ensure_ascii=False),
        )
        for mention in mentions
        if not mention.entity_id and mention.evidence
    ]
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM unresolved_references WHERE document_name = ?",
            (document_name,),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO unresolved_references
                (document_name, file_hash, mention_id, mention_text, entity_name,
                 entity_category, mention_type, evidence, confidence, attributes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def delete_unresolved_references(document_name: str) -> None:
    """Delete unresolved mention evidence for one exact document."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM unresolved_references WHERE document_name = ?",
            (document_name,),
        )


def save_graph_evidence(
    document_id: str,
    document_name: str,
    file_hash: str,
    result: Any,
) -> None:
    """Atomically replace exact evidence/provenance records for accepted graph facts."""
    from graph_rag.config import settings

    rows = []
    for object_type, objects in (
        ("entity", getattr(result, "entities", [])),
        ("relationship", getattr(result, "relations", [])),
        ("topic", getattr(result, "topics", [])),
        ("mention", getattr(result, "mentions", [])),
    ):
        for item in objects:
            evidence = str(getattr(item, "evidence", "") or "")
            if not evidence:
                continue
            provenance = getattr(item, "provenance", {}) or {}
            if object_type == "relationship":
                object_id = "|".join(
                    (
                        str(getattr(item, "source_entity_id", "")),
                        str(getattr(item, "relation_type", "")),
                        str(getattr(item, "target_entity_id", "")),
                    )
                )
            elif object_type == "mention":
                object_id = str(
                    getattr(item, "mention_id", "")
                    or getattr(item, "text", "")
                )
            else:
                object_id = str(
                    getattr(item, "entity_id", "")
                    or getattr(item, "name", "")
                )
            mention_text = (
                str(getattr(item, "text", "") or "")
                if object_type == "mention"
                else ""
            )
            evidence_records = provenance.get("evidence_records") or []
            if not evidence_records:
                evidence_records = [
                    {
                        "evidence": evidence,
                        "confidence": getattr(item, "confidence", 0.0),
                        **provenance,
                    }
                ]
            for record in evidence_records:
                record_evidence = str(record.get("evidence") or evidence)
                record_provenance = {**provenance, **record}
                if not record_evidence:
                    continue
                rows.append(
                    (
                        document_id,
                        document_name,
                        file_hash,
                        object_type,
                        object_id,
                        mention_text,
                        record_evidence,
                        record_provenance.get("page_number"),
                        str(record_provenance.get("sheet_name") or ""),
                        str(record_provenance.get("section") or ""),
                        json.dumps(
                            record_provenance.get("source_offsets") or [],
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            record_provenance.get("bounding_boxes") or [],
                            ensure_ascii=False,
                        ),
                        float(
                            record.get("confidence", getattr(item, "confidence", 0.0))
                            or 0.0
                        ),
                        settings.EXTRACTION_MODEL,
                        settings.EXTRACTION_VERSION,
                    )
                )
    with get_db_connection() as conn:
        conn.execute("DELETE FROM graph_evidence WHERE document_id = ?", (document_id,))
        if rows:
            conn.executemany(
                """
                INSERT INTO graph_evidence
                (document_id, document_name, file_hash, object_type, object_id,
                 mention_text, evidence_text, page_number, sheet_name, section,
                 source_offsets_json, bounding_boxes_json, confidence,
                 extraction_model, extraction_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def save_extraction_diagnostics(
    document_id: str,
    document_name: str,
    file_hash: str,
    diagnostics: List[Dict[str, Any]],
) -> None:
    """Replace non-blocking ambiguity and rejection diagnostics for one document."""
    rows = [
        (
            document_id,
            document_name,
            file_hash,
            str(item.get("diagnostic_type") or "unresolved"),
            json.dumps(item, ensure_ascii=False, default=str),
            str(item.get("evidence") or ""),
            item.get("confidence"),
        )
        for item in diagnostics
    ]
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM extraction_diagnostics WHERE document_id = ?",
            (document_id,),
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO extraction_diagnostics
                (document_id, document_name, file_hash, diagnostic_type,
                 details_json, evidence, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def delete_document_evidence(document_id: str) -> None:
    """Delete evidence and diagnostics for one document identifier."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM graph_evidence WHERE document_id = ?", (document_id,))
        conn.execute(
            "DELETE FROM extraction_diagnostics WHERE document_id = ?",
            (document_id,),
        )


def _claim_extraction_lock(
    document_hash: str,
    owner_token: str,
    lease_seconds: float,
) -> bool:
    now = time.time()
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO extraction_locks (document_hash, owner_token, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(document_hash) DO UPDATE SET
                owner_token = excluded.owner_token,
                expires_at = excluded.expires_at
            WHERE extraction_locks.expires_at <= ?
               OR extraction_locks.owner_token = excluded.owner_token
            """,
            (document_hash, owner_token, now + lease_seconds, now),
        )
    return cursor.rowcount == 1


def _release_extraction_lock(document_hash: str, owner_token: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM extraction_locks WHERE document_hash = ? AND owner_token = ?",
            (document_hash, owner_token),
        )


@contextmanager
def extraction_lock(
    document_hash: str,
    *,
    wait_seconds: float = 600.0,
    lease_seconds: float = 180.0,
) -> Generator[None, None, None]:
    """Serialize extraction for identical content across threads and processes."""
    owner_token = uuid.uuid4().hex
    deadline = time.monotonic() + wait_seconds
    while not _claim_extraction_lock(document_hash, owner_token, lease_seconds):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for document extraction lock: {document_hash}"
            )
        time.sleep(0.2)

    stopped = threading.Event()
    lease_lost = threading.Event()

    def renew() -> None:
        interval = max(1.0, lease_seconds / 3)
        while not stopped.wait(interval):
            try:
                if not _claim_extraction_lock(document_hash, owner_token, lease_seconds):
                    lease_lost.set()
                    return
            except sqlite3.Error:
                logger.exception("Could not renew extraction lock for %s", document_hash)
                lease_lost.set()
                return

    thread = threading.Thread(target=renew, daemon=True)
    thread.start()
    try:
        yield
        if lease_lost.is_set():
            raise RuntimeError(
                f"Lost document extraction lock while processing {document_hash}"
            )
    finally:
        stopped.set()
        thread.join(timeout=lease_seconds)
        _release_extraction_lock(document_hash, owner_token)


def has_incomplete_ingestion_run(document_name: str, file_hash: str) -> bool:
    """Whether the latest run for this exact document content needs a retry."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT status
            FROM ingestion_runs
            WHERE document_name = ? AND file_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (document_name, file_hash),
        ).fetchone()
    return bool(row and row["status"] != "completed")


def get_incomplete_ingestion_run(
    file_hash: str,
    mode: str,
) -> Dict[str, Any] | None:
    """Return the latest incomplete run for this hash and ingestion mode."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT run.document_name, run.file_hash, run.status, run.stage
            FROM ingestion_runs AS run
            WHERE run.mode = ?
              AND run.file_hash = ?
              AND run.id = (
                  SELECT MAX(latest.id)
                  FROM ingestion_runs AS latest
                  WHERE latest.mode = run.mode
                    AND latest.document_name = run.document_name
                    AND latest.file_hash = run.file_hash
              )
              AND run.status != 'completed'
            ORDER BY run.id DESC
            LIMIT 1
            """,
            (mode, file_hash),
        ).fetchone()
    return dict(row) if row else None


def start_ingestion_run(document_name: str, file_hash: str, mode: str) -> int:
    """Start or resume one document/hash run and persist its attempt count."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, status
            FROM ingestion_runs
            WHERE mode = ? AND document_name = ? AND file_hash = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (mode, document_name, file_hash),
        ).fetchone()
        if row and row["status"] != "completed":
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'running', stage = 'started',
                    attempt_count = attempt_count + 1, error_message = '',
                    started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP, completed_at = NULL
                WHERE id = ?
                """,
                (row["id"],),
            )
            return int(row["id"])

        cursor = conn.execute(
            """
            INSERT INTO ingestion_runs
            (mode, document_name, file_hash, status, stage)
            VALUES (?, ?, ?, 'running', 'started')
            """,
            (mode, document_name, file_hash),
        )
        return int(cursor.lastrowid)


def update_ingestion_run(run_id: int, stage: str) -> None:
    """Persist the latest completed ingestion stage."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE ingestion_runs
            SET stage = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND status = 'running'
            """,
            (stage, run_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Ingestion run {run_id} is not active")


def finish_ingestion_run(
    run_id: int,
    *,
    error_message: str = "",
) -> None:
    """Mark a run completed or failed after all current steps have settled."""
    completed = not error_message
    with get_db_connection() as conn:
        if completed:
            cursor = conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'completed', stage = 'completed',
                    error_message = '', updated_at = CURRENT_TIMESTAMP,
                    completed_at = CURRENT_TIMESTAMP
                WHERE document_name = (
                          SELECT document_name FROM ingestion_runs WHERE id = ?
                      )
                  AND file_hash = (
                          SELECT file_hash FROM ingestion_runs WHERE id = ?
                      )
                  AND status != 'completed'
                """,
                (run_id, run_id),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'failed', stage = 'failed',
                    error_message = ?, updated_at = CURRENT_TIMESTAMP,
                    completed_at = NULL
                WHERE id = ? AND status = 'running'
                """,
                (error_message[:4000], run_id),
            )
        if cursor.rowcount != 1:
            if completed and cursor.rowcount > 1:
                return
            raise RuntimeError(f"Ingestion run {run_id} is not active")
