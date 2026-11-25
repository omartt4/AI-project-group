"""Focused offline tests for deterministic retrieval metrics."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.evaluation.retrieval_metrics import (  # noqa: E402
    evaluate_question,
    parse_page_spec,
    summarize_results,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_parse_page_ranges(self) -> None:
        self.assertEqual(parse_page_spec("1,4-6,74"), [1, 4, 5, 6, 74])

    def test_rank_metrics_and_graph_recall(self) -> None:
        gold = {
            "question_id": "q1",
            "question": "test",
            "answerable": True,
            "expected_document": "آفاق اقتصادية العدد 57.pdf",
            "relevant_pdf_pages": [43, 44],
            "expected_entities": ["بريكس بلس"],
            "expected_relations": ["SHARE_INCREASES"],
        }
        chunks = [
            {"metadata": {"document_name": "other.pdf", "page": 1}},
            {"metadata": {"document_name": "آفاق اقتصادية العدد 57.pdf", "page": 43}},
            {"metadata": {"document_name": "آفاق اقتصادية العدد 57.pdf", "page": 44}},
        ]
        row = evaluate_question(
            gold,
            chunks=chunks,
            concepts=[{"name": "مجموعة بريكس بلس"}],
            graph_facts=[{"relation": "SHARE_INCREASES"}],
            ks=(1, 3),
        )
        self.assertEqual(row["hit_rate@1"], 0.0)
        self.assertEqual(row["hit_rate@3"], 1.0)
        self.assertEqual(row["recall@3"], 1.0)
        self.assertAlmostEqual(row["precision@3"], 2 / 3)
        self.assertEqual(row["mrr"], 0.5)
        self.assertEqual(row["entity_recall"], 1.0)
        self.assertEqual(row["relation_recall"], 1.0)

    def test_false_retrieval_for_unanswerable_question(self) -> None:
        gold = {
            "question_id": "q2",
            "question": "not in corpus",
            "answerable": False,
        }
        weak = evaluate_question(
            gold,
            chunks=[{"similarity_score": 0.1, "metadata": {"page": 1}}],
            min_similarity=0.35,
        )
        strong = evaluate_question(
            gold,
            chunks=[{"similarity_score": 0.8, "metadata": {"page": 1}}],
            min_similarity=0.35,
        )
        self.assertFalse(weak["false_retrieval"])
        self.assertTrue(strong["false_retrieval"])

    def test_summary_does_not_turn_unavailable_into_zero(self) -> None:
        summary = summarize_results(
            [
                {
                    "answerable": True,
                    "mrr": 1.0,
                    "entity_recall": None,
                    "relation_recall": None,
                    "hit_rate@3": 1.0,
                    "precision@3": 1 / 3,
                    "recall@3": 1.0,
                    "false_retrieval": None,
                }
            ],
            ks=(3,),
        )
        self.assertIsNone(summary["entity_recall"])
        self.assertEqual(summary["entity_recall_evaluated_questions"], 0)


if __name__ == "__main__":
    unittest.main()
