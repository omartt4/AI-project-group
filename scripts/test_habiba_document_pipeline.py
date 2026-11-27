"""Focused offline tests for Habiba's OCR and graph-extraction tasks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from graph_rag.graph.entity_extractor import ExtractedEntity, ExtractedRelation
from graph_rag.graph.entity_resolution import EntityResolver, normalize_entity_name
from graph_rag.graph.topic_taxonomy import TopicTaxonomy
from graph_rag.ingestion.ocr import (
    FallbackOCRProvider,
    OCRResult,
    PaddleOCRProvider,
    SuryaOCRProvider,
)
from graph_rag.ingestion.spacy_candidates import SpacyCandidateExtractor
from benchmark_ocr import character_error_rate, word_error_rate


class HabibaDocumentPipelineTests(unittest.TestCase):
    def test_arabic_stanza_dependency_rows_and_noun_phrases(self) -> None:
        class Word:
            def __init__(self, word_id, text, upos, head, deprel):
                self.id = word_id
                self.text = text
                self.upos = upos
                self.head = head
                self.deprel = deprel

        class Sentence:
            text = "يدير أحمد شركة تقنية كبيرة."
            words = [
                Word(1, "يدير", "VERB", 0, "root"),
                Word(2, "أحمد", "PROPN", 1, "nsubj"),
                Word(3, "شركة", "NOUN", 1, "obj"),
                Word(4, "تقنية", "ADJ", 3, "amod"),
                Word(5, "كبيرة", "ADJ", 3, "amod"),
            ]

        class Document:
            sentences = [Sentence()]

        extractor = SpacyCandidateExtractor()
        extractor._arabic_stanza_pipeline = lambda text: Document()
        result = extractor._arabic_dependency_rows(Sentence.text)
        self.assertEqual(len(result["dependencies"]), 5)
        self.assertEqual(result["dependencies"][1]["head"], "يدير")
        self.assertIn("شركة تقنية كبيرة", result["noun_phrases"])

    def test_paddle_v3_result_preserves_text_confidence_and_boxes(self) -> None:
        result = PaddleOCRProvider._parse_results(
            [
                {
                    "rec_texts": ["الاقتصاد العالمي", "Global economy"],
                    "rec_scores": [0.95, 0.91],
                    "rec_polys": [
                        [[10, 20], [110, 20], [110, 40], [10, 40]],
                        [[10, 50], [150, 50], [150, 70], [10, 70]],
                    ],
                }
            ]
        )
        self.assertEqual(result.text, "الاقتصاد العالمي\nGlobal economy")
        self.assertAlmostEqual(result.confidence or 0.0, 0.93)
        self.assertEqual(result.bounding_boxes[0]["x"], 10)
        self.assertEqual(result.bounding_boxes[0]["width"], 100)

    def test_ocr_fallback_uses_second_available_provider(self) -> None:
        class FailedProvider:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def extract(image_bytes: bytes, language: str) -> OCRResult:
                raise RuntimeError("broken backend")

        class WorkingProvider:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def extract(image_bytes: bytes, language: str) -> OCRResult:
                return OCRResult("fallback text", metadata={"engine": "working"})

        result = FallbackOCRProvider(
            [FailedProvider(), WorkingProvider()]
        ).extract(b"image", "eng")
        self.assertEqual(result.text, "fallback text")
        self.assertEqual(result.metadata["engine"], "working")

    def test_surya_stdout_preserves_tables_confidence_and_boxes(self) -> None:
        stdout = (
            '{"page.png":[{"blocks":['
            '{"label":"Text","html":"Arabic text","bbox":[1,2,11,12],'
            '"confidence":0.9},'
            '{"label":"Table","html":"<table><tr><td>A</td></tr></table>",'
            '"bbox":[2,13,20,30],"confidence":0.8}'
            ']}]}'
        )
        provider = SuryaOCRProvider(command="surya_ocr")
        with (
            patch.object(provider, "is_available", return_value=True),
            patch(
                "graph_rag.ingestion.ocr.subprocess.run",
                return_value=SimpleNamespace(stdout=stdout),
            ),
        ):
            result = provider.extract(b"fake-image", "ara")
        self.assertIn("Arabic text", result.text)
        self.assertEqual(result.bounding_boxes[0]["width"], 10)
        self.assertEqual(len(result.tables), 1)
        self.assertAlmostEqual(result.confidence or 0.0, 0.85)

    def test_cer_and_wer_are_deterministic(self) -> None:
        self.assertEqual(character_error_rate("abc", "abc"), 0.0)
        self.assertAlmostEqual(character_error_rate("abc", "adc"), 1 / 3)
        self.assertAlmostEqual(word_error_rate("one two", "one three"), 0.5)

    def test_arabic_name_normalization_handles_common_variants(self) -> None:
        self.assertEqual(
            normalize_entity_name("أَحْمَد الـمَصرى"),
            normalize_entity_name("احمد المصري"),
        )
        self.assertEqual(
            normalize_entity_name("عليّ"),
            normalize_entity_name("على"),
        )

    def test_entity_resolution_uses_relationship_context(self) -> None:
        class FakeGraph:
            def __init__(self) -> None:
                self.last_context = []

            def find_entity_candidates(self, **kwargs):
                self.last_context = kwargs.get("relationship_context", [])
                if self.last_context:
                    return [
                        {
                            "entity_id": "person:khaled",
                            "name": "خالد محمود",
                            "category": "Person",
                            "description": "",
                            "aliases": ["خالد"],
                            "attributes": {"role": "مدير", "organization": "مايكروسوفت"},
                        }
                    ]
                return []

        class FakeVectorStore:
            @staticmethod
            def search_entity_vectors_batch(queries):
                return [[] for _ in queries]

        person = ExtractedEntity(
            name="المدير",
            category="Person",
            entity_id="local-person",
        )
        company = ExtractedEntity(
            name="مايكروسوفت",
            category="Organization",
            entity_id="local-company",
        )
        relation = ExtractedRelation(
            source="المدير",
            target="مايكروسوفت",
            relation_type="MANAGES",
            source_category="Person",
            target_category="Organization",
            source_entity_id="local-person",
            target_entity_id="local-company",
        )
        graph = FakeGraph()
        resolved, _, _, id_map = EntityResolver(
            graph,
            FakeVectorStore(),
        ).resolve([person, company], [relation])
        self.assertTrue(graph.last_context)
        self.assertEqual(id_map["local-person"], "person:khaled")
        self.assertEqual(resolved[0].name, "خالد محمود")

    def test_topic_hierarchy_rejects_cycles_and_keeps_multiple_parents(self) -> None:
        taxonomy = TopicTaxonomy(minimum_confidence=0.8)
        diagnostics = []
        assignments = taxonomy.assign(
            proposals=[
                {
                    "name": "Topic A",
                    "parent_names": ["Topic B"],
                    "evidence": "Topic A belongs to Topic B",
                    "confidence": 0.95,
                },
                {
                    "name": "Topic B",
                    "parent_names": ["Topic A"],
                    "evidence": "Topic B belongs to Topic A",
                    "confidence": 0.95,
                },
                {
                    "name": "Multimodal Learning",
                    "parent_names": ["Machine Learning", "Computer Vision"],
                    "evidence": "Multimodal learning combines both fields",
                    "confidence": 0.95,
                },
            ],
            diagnostics=diagnostics,
        )
        multimodal = next(
            item for item in assignments if item.name == "Multimodal Learning"
        )
        self.assertEqual(
            set(multimodal.parent_names),
            {"Machine Learning", "Computer Vision"},
        )
        self.assertTrue(
            any(row["diagnostic_type"] == "cyclic_topic_parent" for row in diagnostics)
        )
        cyclic = [
            item for item in assignments if item.name in {"Topic A", "Topic B"}
        ]
        self.assertTrue(all(item.level == "top_level" for item in cyclic))


if __name__ == "__main__":
    unittest.main()
