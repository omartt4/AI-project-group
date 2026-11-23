"""Persist and summarize token-based costs calculated from a local rate card."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

from graph_rag.database.db import get_db_connection
from graph_rag.llm.pricing import PRICING_VERSION, calculate_cost_usd

logger = logging.getLogger(__name__)


class CostTracker:
    """Store per-request model usage, latency, and outcome."""

    def record(
        self,
        *,
        role: str,
        operation: str,
        model: str,
        endpoint: str,
        usage: Dict[str, Any],
        request_id: str,
        run_id: str | None = None,
        user_id: int | None = None,
        latency_ms: int | None = None,
        status: str = "success",
        retry_count: int | None = None,
        cache_status: str = "miss",
        error_message: str | None = None,
    ) -> None:
        prompt_tokens = self._optional_int(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        completion_tokens = self._optional_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        total_tokens = self._optional_int(usage.get("total_tokens"))
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        provider_cost_usd = self._optional_float(usage.get("cost"))
        calculation = calculate_cost_usd(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost_usd = (
            float(calculation.cost_usd)
            if calculation.cost_usd is not None
            else None
        )
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO model_usage
                    (role, operation, model, endpoint, prompt_tokens,
                    completion_tokens, total_tokens, cost_usd,
                    provider_cost_usd, input_rate_usd_per_million,
                    output_rate_usd_per_million, cost_method, pricing_version,
                    request_id, run_id, user_id, latency_ms, status, retry_count,
                    cache_status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        provider_cost_usd,
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
                        PRICING_VERSION if calculation.input_usd_per_million is not None else None,
                        request_id,
                        run_id,
                        user_id,
                        latency_ms,
                        status,
                        retry_count,
                        cache_status,
                        error_message[:1000] if error_message else None,
                    ),
                )
            logger.info(
                "Model request: role=%s operation=%s model=%s status=%s "
                "request_id=%s run_id=%s input_tokens=%s output_tokens=%s "
                "total_tokens=%s input_rate_per_million=%s "
                "output_rate_per_million=%s cost_usd=%s cost_method=%s "
                "provider_cost_usd=%s latency_ms=%s retry_count=%s "
                "cache_status=%s",
                role,
                operation,
                model,
                status,
                request_id,
                run_id,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                calculation.input_usd_per_million,
                calculation.output_usd_per_million,
                cost_usd,
                calculation.method,
                provider_cost_usd,
                latency_ms,
                retry_count,
                cache_status,
            )
        except Exception as exc:
            logger.warning("Could not persist model usage: %s", exc)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0, parsed)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0 else None

    def last_id(self) -> int:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS id FROM model_usage"
            ).fetchone()
        return int(row["id"] if row else 0)

    def summary(self, since_id: int = 0) -> Dict[str, Any]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT role,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS requests,
                       SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(SUM(provider_cost_usd), 0.0) AS provider_cost_usd,
                       SUM(CASE WHEN cost_usd IS NULL
                           THEN 1 ELSE 0 END) AS unknown_cost_requests,
                       SUM(CASE WHEN total_tokens IS NULL
                           THEN 1 ELSE 0 END) AS unknown_token_requests
                FROM model_usage
                WHERE id > ?
                GROUP BY role
                ORDER BY cost_usd DESC
                """,
                (since_id,),
            ).fetchall()
        breakdown = [dict(row) for row in rows]
        return {
            "requests": sum(int(row["requests"] or 0) for row in breakdown),
            "failed_requests": sum(
                int(row["failed_requests"] or 0) for row in breakdown
            ),
            "total_tokens": sum(int(row["total_tokens"] or 0) for row in breakdown),
            "cost_usd": sum(float(row["cost_usd"] or 0.0) for row in breakdown),
            "provider_cost_usd": sum(
                float(row["provider_cost_usd"] or 0.0) for row in breakdown
            ),
            "unknown_cost_requests": sum(
                int(row["unknown_cost_requests"] or 0) for row in breakdown
            ),
            "unknown_token_requests": sum(
                int(row["unknown_token_requests"] or 0) for row in breakdown
            ),
            "breakdown": breakdown,
        }

    def recent(self, limit: int = 10) -> list[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, operation, model, prompt_tokens,
                       completion_tokens, total_tokens, cost_usd,
                       provider_cost_usd, input_rate_usd_per_million,
                       output_rate_usd_per_million, cost_method, pricing_version,
                       request_id, run_id, user_id, latency_ms,
                       status, retry_count, cache_status, error_message, created_at
                FROM model_usage
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


cost_tracker = CostTracker()
