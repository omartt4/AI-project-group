"""
Evaluation Service module.

Executes live RAG evaluations, persists scores to the SQLite evaluation_logs
table, and exposes aggregation functions for the evaluation dashboard.

Changes vs. the original:
  * run_tag column: separates test runs (e.g. "graph", "vector_only") from
    normal chat usage so the averages are not mixed.
  * Automatic schema migration (adds run_tag if the column is missing).
  * Per-metric timeout + error handling: a hanging or failing LLM judge call
    no longer blocks or crashes the whole run.
  * Full context is stored (up to MAX_CONTEXT_CHARS) instead of 1500 chars.
  * Persist failures are logged as errors and can optionally be raised.
  * New helpers: get_run_tags(), get_summary_by_run(), compare_runs().
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from graph_rag.config import settings
from graph_rag.database.db import get_db_connection
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.evaluation.metrics import (
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000       # stored context length (was 1500)
METRIC_TIMEOUT_S = 120.0       # per-metric LLM call timeout


@dataclass
class EvaluationRecord:
    """Full evaluation assessment for a single query response."""
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    faithfulness_reasoning: str
    answer_relevance_reasoning: str
    context_relevance_reasoning: str
    latency: LatencyMetrics


def _safe_metric(
    name: str,
    fn: Callable[..., Tuple[float, str]],
    *args,
    timeout: float = METRIC_TIMEOUT_S,
) -> Tuple[float, str]:
    """Run one metric with a timeout. Never raises: returns (0.0, reason) on failure.

    Note: a timed-out call cannot be killed (Python threads), it is simply
    abandoned, so the evaluation moves on instead of hanging forever.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        logger.error("Metric '%s' timed out after %.0fs", name, timeout)
        return 0.0, f"metric_error: {name} timed out after {timeout:.0f}s"
    except Exception as e:  # judge failure must not kill the run
        logger.error("Metric '%s' failed: %s", name, e)
        return 0.0, f"metric_error: {name} failed: {e}"
    finally:
        pool.shutdown(wait=False)


class EvaluationService:
    """Handles evaluation logging and dashboard metrics generation."""

    def __init__(self) -> None:
        self._schema_checked = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Add the run_tag column to evaluation_logs if it does not exist."""
        if self._schema_checked:
            return
        try:
            with get_db_connection() as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(evaluation_logs)").fetchall()]
                if cols and "run_tag" not in cols:
                    conn.execute("ALTER TABLE evaluation_logs ADD COLUMN run_tag TEXT")
                    logger.info("Added run_tag column to evaluation_logs")
            self._schema_checked = True
        except Exception as e:
            logger.error("Could not verify/migrate evaluation_logs schema: %s", e)

    # ------------------------------------------------------------------
    # Evaluate one turn
    # ------------------------------------------------------------------
    def evaluate_turn(
        self,
        question: str,
        answer: str,
        context: str,
        message_id: Optional[int] = None,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        latency: Optional[LatencyMetrics] = None,
        run_tag: Optional[str] = None,
        raise_on_persist_error: bool = False,
    ) -> EvaluationRecord:
        """Run the full evaluation suite on an answered turn and persist it.

        run_tag: None for normal chat usage; use e.g. "graph" or "vector_only"
                 for benchmark runs so they are kept apart in the summaries.
        """
        self._ensure_schema()
        lat = latency or LatencyMetrics()

        with StageTimer() as eval_timer:
            faith_score, faith_reason = _safe_metric(
                "faithfulness", evaluate_faithfulness, question, answer, context
            )
            relevance_score, relevance_reason = _safe_metric(
                "answer_relevance", evaluate_answer_relevance, question, answer
            )
            ctx_score, ctx_reason = _safe_metric(
                "context_relevance", evaluate_context_relevance, question, context
            )

        lat.eval_ms = eval_timer.elapsed_ms
        lat.total_ms = lat.retrieval_ms + lat.generation_ms + lat.eval_ms

        eval_record = EvaluationRecord(
            faithfulness=faith_score,
            answer_relevance=relevance_score,
            context_relevance=ctx_score,
            faithfulness_reasoning=faith_reason,
            answer_relevance_reasoning=relevance_reason,
            context_relevance_reasoning=ctx_reason,
            latency=lat,
        )

        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO evaluation_logs
                    (message_id, session_id, user_id, question, answer, context_chunks,
                     faithfulness, answer_relevance, context_relevance,
                     retrieval_ms, generation_ms, eval_ms, total_ms,
                     eval_reasoning, run_tag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_id,
                        user_id,
                        question,
                        answer,
                        (context or "")[:MAX_CONTEXT_CHARS],
                        faith_score,
                        relevance_score,
                        ctx_score,
                        lat.retrieval_ms,
                        lat.generation_ms,
                        lat.eval_ms,
                        lat.total_ms,
                        json.dumps({
                            "faithfulness": faith_reason,
                            "answer_relevance": relevance_reason,
                            "context_relevance": ctx_reason,
                        }),
                        run_tag,
                    ),
                )
        except Exception as e:
            logger.error("Could not persist evaluation log to database: %s", e)
            if raise_on_persist_error:
                raise

        logger.info(
            "Evaluation scores [%s]: faithfulness=%.3f answer_relevance=%.3f "
            "context_relevance=%.3f retrieval_ms=%d generation_ms=%d "
            "eval_ms=%d total_ms=%d",
            run_tag or "live",
            faith_score,
            relevance_score,
            ctx_score,
            lat.retrieval_ms,
            lat.generation_ms,
            lat.eval_ms,
            lat.total_ms,
        )
        return eval_record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_session_eval_logs(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieve evaluation logs for a given chat session."""
        self._ensure_schema()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, message_id, question, answer, faithfulness,
                       answer_relevance, context_relevance, retrieval_ms,
                       generation_ms, eval_ms, total_ms, eval_reasoning,
                       run_tag, created_at
                FROM evaluation_logs
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_logs(
        self, limit: int = 50, run_tag: Optional[str] = None, all_runs: bool = True
    ) -> List[Dict[str, Any]]:
        """Most recent evaluation logs for the live view.

        all_runs=True (default, same as the old behaviour) returns every row.
        Set all_runs=False to filter by run_tag (None = normal chat only).
        """
        self._ensure_schema()
        where, params = self._run_filter(run_tag, all_runs)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT e.*, u.username, s.title AS session_title
                FROM evaluation_logs e
                LEFT JOIN users u ON e.user_id = u.id
                LEFT JOIN chat_sessions s ON e.session_id = s.id
                {where}
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _run_filter(run_tag: Optional[str], all_runs: bool) -> Tuple[str, tuple]:
        if all_runs:
            return "", ()
        if run_tag is None:
            return "WHERE e.run_tag IS NULL", ()
        return "WHERE e.run_tag = ?", (run_tag,)

    @staticmethod
    def _empty_summary() -> Dict[str, Any]:
        return {
            "total_evaluations": 0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevance": 0.0,
            "avg_context_relevance": 0.0,
            "avg_retrieval_ms": 0,
            "avg_generation_ms": 0,
            "avg_total_ms": 0,
        }

    _SUMMARY_SELECT = """
        COUNT(*) as total_evaluations,
        ROUND(AVG(faithfulness), 2) as avg_faithfulness,
        ROUND(AVG(answer_relevance), 2) as avg_answer_relevance,
        ROUND(AVG(context_relevance), 2) as avg_context_relevance,
        ROUND(AVG(retrieval_ms), 0) as avg_retrieval_ms,
        ROUND(AVG(generation_ms), 0) as avg_generation_ms,
        ROUND(AVG(total_ms), 0) as avg_total_ms
    """

    def get_overall_summary(
        self, run_tag: Optional[str] = None, all_runs: bool = True
    ) -> Dict[str, Any]:
        """Aggregate summary metrics.

        all_runs=True (default, same as the old behaviour): every row.
        all_runs=False: only rows with this run_tag (None = normal chat usage).
        """
        self._ensure_schema()
        where, params = self._run_filter(run_tag, all_runs)
        where = where.replace("e.", "")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._SUMMARY_SELECT} FROM evaluation_logs {where}", params
            )
            row = cursor.fetchone()
            if row and row["total_evaluations"]:
                return dict(row)
            return self._empty_summary()

    def get_run_tags(self) -> List[str]:
        """All distinct run tags used so far (benchmark runs only)."""
        self._ensure_schema()
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_tag FROM evaluation_logs "
                "WHERE run_tag IS NOT NULL ORDER BY run_tag"
            ).fetchall()
            return [r[0] for r in rows]

    def get_summary_by_run(self) -> Dict[str, Dict[str, Any]]:
        """Summary per run tag; normal chat usage appears under 'live'."""
        self._ensure_schema()
        with get_db_connection() as conn:
            rows = conn.execute(
                f"SELECT COALESCE(run_tag, 'live') AS tag, {self._SUMMARY_SELECT} "
                "FROM evaluation_logs GROUP BY COALESCE(run_tag, 'live')"
            ).fetchall()
            return {r["tag"]: {k: r[k] for k in r.keys() if k != "tag"} for r in rows}

    def compare_runs(self, tag_a: str, tag_b: str) -> Dict[str, Any]:
        """Side-by-side of two run tags; delta = A - B (positive = A better,
        except for the *_ms latency fields where lower is better)."""
        a = self.get_overall_summary(tag_a, all_runs=False)
        b = self.get_overall_summary(tag_b, all_runs=False)
        delta = {
            k: round((a[k] or 0) - (b[k] or 0), 3)
            for k in a
            if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
        }
        return {tag_a: a, tag_b: b, "delta_a_minus_b": delta}


eval_service = EvaluationService()