"""Persist and summarize the usage and cost reported by OpenRouter."""

from __future__ import annotations

import logging
from typing import Any, Dict

from graph_rag.database.db import get_db_connection

logger = logging.getLogger(__name__)


class CostTracker:
    """Store one row per successful paid model request."""

    def record(
        self,
        *,
        role: str,
        operation: str,
        model: str,
        endpoint: str,
        usage: Dict[str, Any],
    ) -> None:
        if not usage:
            return
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        )
        raw_cost = usage.get("cost")
        cost_usd = float(raw_cost) if raw_cost is not None else None
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO model_usage
                    (role, operation, model, endpoint, prompt_tokens,
                     completion_tokens, total_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        role,
                        operation,
                        model,
                        endpoint,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        cost_usd,
                    ),
                )
            cost_label = f"${cost_usd:.8f}" if cost_usd is not None else "unknown"
            logger.info(
                "Model usage: role=%s operation=%s model=%s "
                "prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%s",
                role,
                operation,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_label,
            )
        except Exception as exc:
            logger.warning("Could not persist model usage: %s", exc)

    def last_id(self) -> int:
        try:
            with get_db_connection() as conn:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) AS id FROM model_usage").fetchone()
                return int(row["id"] if row else 0)
        except Exception:
            return 0

    def summary(self, since_id: int = 0) -> Dict[str, Any]:
        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT role, COUNT(*) AS requests,
                           COALESCE(SUM(total_tokens), 0) AS total_tokens,
                           COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                           SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END)
                               AS unknown_cost_requests
                    FROM model_usage
                    WHERE id > ?
                    GROUP BY role
                    ORDER BY cost_usd DESC
                    """,
                    (since_id,),
                ).fetchall()
            breakdown = [dict(row) for row in rows]
            return {
                "requests": sum(int(row["requests"]) for row in breakdown),
                "total_tokens": sum(int(row["total_tokens"]) for row in breakdown),
                "cost_usd": sum(float(row["cost_usd"]) for row in breakdown),
                "unknown_cost_requests": sum(
                    int(row["unknown_cost_requests"]) for row in breakdown
                ),
                "breakdown": breakdown,
            }
        except Exception:
            return {
                "requests": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "unknown_cost_requests": 0,
                "breakdown": [],
            }

    def recent(self, limit: int = 10) -> list[Dict[str, Any]]:
        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT role, operation, model, prompt_tokens,
                           completion_tokens, total_tokens, cost_usd, created_at
                    FROM model_usage
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []


cost_tracker = CostTracker()
