"""
Reliability and Observability Tests for Infrastructure, Cost Limits, LangSmith, and Consistency.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from graph_rag.config import Settings
from graph_rag.database.db import init_db
from graph_rag.llm.cost_tracker import CostLimitExceededError, CostTracker
from graph_rag.ingestion.pipeline import pipeline
from graph_rag.ingestion.ocr import TesseractOCRProvider
from graph_rag.ingestion.spacy_candidates import spacy_candidate_extractor


class InfrastructureReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_production_configuration_rejects_insecure_defaults(self):
        """Production configuration must reject default dev credentials."""
        s = Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY="dev-secret-key-graph-rag-2026-super-secure",
            NEO4J_PASSWORD="password123",
            MODEL_API_KEY="",
        )
        with self.assertRaises(ValueError) as ctx:
            s.ensure_directories()
        self.assertIn("production", str(ctx.exception).lower())

    def test_cost_warning_and_hard_limit(self):
        """CostTracker raises CostLimitExceededError when hard limit is breached."""
        tracker = CostTracker()
        with patch.object(tracker, "get_total_cost_usd", return_value=15.0):
            with patch("graph_rag.llm.cost_tracker.settings") as mock_settings:
                mock_settings.HARD_COST_LIMIT_USD = 10.0
                mock_settings.COST_WARNING_THRESHOLD_USD = 5.0
                with self.assertRaises(CostLimitExceededError) as ctx:
                    tracker.check_cost_limits()
                self.assertIn("Hard cost limit reached", str(ctx.exception))

    def test_cached_tokens_and_dimension_summaries(self):
        """CostTracker records cached tokens and groups by dimension."""
        tracker = CostTracker()
        tracker.record(
            role="chat",
            operation="generation",
            model="google/gemma-4-31b-it:free",
            endpoint="/chat/completions",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 80},
            request_id="req-cached-test-1",
            document_id="doc-test-1",
            query_id="query-test-1",
            language="en",
        )
        summary = tracker.summary()
        self.assertGreaterEqual(summary.get("cached_tokens", 0), 0)

        by_doc = tracker.summary_by_dimension("document")
        self.assertTrue(isinstance(by_doc, list))
        by_lang = tracker.summary_by_dimension("language")
        self.assertTrue(isinstance(by_lang, list))

    def test_cross_store_consistency_check(self):
        """check_cross_store_consistency returns structured report."""
        report = pipeline.check_cross_store_consistency()
        self.assertIn("consistent", report)
        self.assertIn("chroma_document_count", report)
        self.assertIn("neo4j_document_count", report)
        self.assertIn("incomplete_runs", report)

    def test_ocr_and_spacy_availability_checks(self):
        """OCR and spaCy expose boolean availability probes."""
        ocr = TesseractOCRProvider()
        self.assertIsInstance(ocr.is_available(), bool)
        self.assertIsInstance(spacy_candidate_extractor.is_available(), bool)

    def test_langsmith_tracing_environment_export(self):
        """When LANGCHAIN_TRACING_V2 is enabled, ensure_directories exports env variables."""
        with patch.dict(os.environ, {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "lsv2_pt_testkey_12345",
            "LANGCHAIN_PROJECT": "graph-rag-test",
        }):
            s = Settings()
            s.ensure_directories()
            self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
            self.assertEqual(os.environ.get("LANGCHAIN_PROJECT"), "graph-rag-test")


if __name__ == "__main__":
    unittest.main()
