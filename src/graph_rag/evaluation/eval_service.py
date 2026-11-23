"""
Evaluation Service module.

Executes live RAG evaluations, persists scores to SQLite evaluation_logs table,
and exposes aggregation functions for the Live Session Evaluation Dashboard.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from graph_rag.config import settings
from graph_rag.database.db import get_db_connection
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.evaluation.metrics import evaluate_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRecord:
    """Full evaluation assessment for a single query response."""
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    context_relevance: Optional[float]
    faithfulness_reasoning: str
    answer_relevance_reasoning: str
    context_relevance_reasoning: str
    latency: LatencyMetrics


class EvaluationService:
    """Handles evaluation logging and dashboard metrics generation."""

    def evaluate_turn(
        self,
        question: str,
        answer: str,
        context: str,
        message_id: Optional[int] = None,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        latency: Optional[LatencyMetrics] = None,
    ) -> EvaluationRecord:
        """Run full evaluation suite on an answered turn and persist to database."""
        lat = latency or LatencyMetrics()

        with StageTimer() as eval_timer:
            scores = evaluate_all_metrics(question, answer, context)
            faith_score, faith_reason = scores["faithfulness"]
            relevance_score, relevance_reason = scores["answer_relevance"]
            ctx_score, ctx_reason = scores["context_relevance"]

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

        # Log into SQLite
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO evaluation_logs
                    (message_id, session_id, user_id, question, answer, context_chunks,
                     faithfulness, answer_relevance, context_relevance,
                     retrieval_ms, generation_ms, eval_ms, total_ms, eval_reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_id,
                        user_id,
                        question,
                        answer,
                        context[:1500],
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
                    ),
                )
        except Exception as e:
            logger.warning(f"Could not persist evaluation log to database: {e}")

        logger.info(
            "Evaluation scores: faithfulness=%s answer_relevance=%s "
            "context_relevance=%s retrieval_ms=%d generation_ms=%d "
            "eval_ms=%d total_ms=%d",
            f"{faith_score:.3f}" if faith_score is not None else "unavailable",
            f"{relevance_score:.3f}" if relevance_score is not None else "unavailable",
            f"{ctx_score:.3f}" if ctx_score is not None else "unavailable",
            lat.retrieval_ms,
            lat.generation_ms,
            lat.eval_ms,
            lat.total_ms,
        )
        return eval_record

    def get_session_eval_logs(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieve evaluation logs for a given chat session."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, message_id, question, answer, faithfulness,
                       answer_relevance, context_relevance, retrieval_ms,
                       generation_ms, eval_ms, total_ms, eval_reasoning, created_at
                FROM evaluation_logs
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve most recent evaluation logs across all sessions for live view."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.*, u.username, s.title AS session_title
                FROM evaluation_logs e
                LEFT JOIN users u ON e.user_id = u.id
                LEFT JOIN chat_sessions s ON e.session_id = s.id
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_overall_summary(self) -> Dict[str, Any]:
        """Aggregate summary metrics across all sessions."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN faithfulness IS NOT NULL
                                  OR answer_relevance IS NOT NULL
                                  OR context_relevance IS NOT NULL
                             THEN 1 ELSE 0 END) as scored_evaluations,
                    ROUND(AVG(faithfulness), 2) as avg_faithfulness,
                    ROUND(AVG(answer_relevance), 2) as avg_answer_relevance,
                    ROUND(AVG(context_relevance), 2) as avg_context_relevance,
                    ROUND(AVG(retrieval_ms), 0) as avg_retrieval_ms,
                    ROUND(AVG(generation_ms), 0) as avg_generation_ms,
                    ROUND(AVG(total_ms), 0) as avg_total_ms
                FROM evaluation_logs
                """
            )
            row = cursor.fetchone()
            if row and row["total_evaluations"]:
                return dict(row)
            return {
                "total_evaluations": 0,
                "scored_evaluations": 0,
                "avg_faithfulness": None,
                "avg_answer_relevance": None,
                "avg_context_relevance": None,
                "avg_retrieval_ms": 0,
                "avg_generation_ms": 0,
                "avg_total_ms": 0,
            }


eval_service = EvaluationService()
