"""Fast unit checks for the graph-redesign foundations.

These checks do not call OpenRouter, ChromaDB, or Neo4j.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.graph.entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedMention,
    GraphExtractionResult,
)
from graph_rag.graph.entity_resolution import EntityResolver, PaperEntityRegistry
from graph_rag.graph.graph_builder import GraphBuilder
from graph_rag.graph.type_normalization import TypeNormalizer
from graph_rag.graph.topic_taxonomy import topic_taxonomy
from graph_rag.config import settings
from graph_rag.database import db as database
from graph_rag.embeddings.vector_store import VectorStore
from graph_rag.ingestion.metadata_manager import MetadataManager
from graph_rag.ingestion.document_structure import SourceBlock, StructuredDocumentInput
from graph_rag.retriever.query_engine import build_prompts
import graph_rag.ingestion.pipeline as pipeline_module
from graph_rag.ingestion.pipeline import IngestionPipeline
from graph_rag.llm.model_client import (
    CloudModelClient,
    chat_model_client,
    embedding_model_client,
    evaluation_model_client,
    extraction_model_client,
)


class FakeModelClient:
    def generate(self, *args, **kwargs):
        return """{
          "entities": [
            {"entity_id": "", "name": "Ahmed Hassan", "category": "Person",
             "description": "CEO of Example Company", "aliases": ["Ahmed"],
             "attributes": {"role": "CEO", "organization": "Example Company"},
             "evidence": "Ahmed is the CEO of Example Company.", "confidence": 0.99},
            {"entity_id": "", "name": "Example Company", "category": "Organization",
             "description": "An example company", "aliases": [], "attributes": {},
             "evidence": "Ahmed is the CEO of Example Company.", "confidence": 0.99}
          ],
          "mentions": [
            {"text": "Ahmed", "entity_id": "", "entity_name": "Ahmed Hassan",
             "entity_category": "Person", "mention_type": "name",
             "evidence": "Ahmed is the CEO of Example Company.", "confidence": 0.99}
          ],
          "relations": [
            {"source_entity_id": "", "source": "Ahmed Hassan", "source_category": "Person",
             "target_entity_id": "", "target": "Example Company", "target_category": "Organization",
             "relation_type": "HOLDS_ROLE", "description": "Ahmed is CEO",
             "evidence": "Ahmed is the CEO of Example Company.", "confidence": 0.98},
            {"source": "Ahmed Hassan", "target": "Example Company",
             "relation_type": "UNSAFE_DYNAMIC_TYPE", "confidence": 1.0}
          ]
        }"""


class FakeCloudClient(CloudModelClient):
    def _get_chat_model(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(
            lambda _, **kwargs: AIMessage(content="cloud response")
        )

    def _get_embedding_model(self):
        class FakeEmbeddingAPI:
            @staticmethod
            def create(input, **kwargs):
                return {
                    "data": [
                        {
                            "index": index,
                            "embedding": [1.0, 0.0]
                            if index == 0
                            else [0.0, 1.0],
                        }
                        for index, _ in enumerate(input)
                    ],
                    "usage": {
                        "prompt_tokens": len(input),
                        "completion_tokens": 0,
                        "total_tokens": len(input),
                    },
                }

        return SimpleNamespace(
            chunk_size=1000,
            _invocation_params={},
            _ensure_sync_client_available=lambda: None,
            client=FakeEmbeddingAPI(),
        )


def test_topics() -> None:
    assert topic_taxonomy.assign("NLP; Medical Informatics") == []

    proposals = topic_taxonomy.assign(
        proposals=[
            {
                "name": "Machine Learning",
                "evidence": "The report focuses on machine learning.",
                "confidence": 0.96,
            },
            {
                "name": "Neural Text Classification",
                "parent_names": ["Machine Learning"],
                "evidence": "Neural text classification is the evaluated task.",
                "confidence": 0.93,
            },
            {
                "name": "Weak Topic",
                "evidence": "A weakly supported phrase.",
                "confidence": 0.3,
            },
            {
                "name": "OpenAI",
                "evidence": "OpenAI released the system.",
                "confidence": 0.99,
            },
        ],
        entity_names=["OpenAI"],
    )
    proposed = {item.name: item for item in proposals}
    assert proposed["Machine Learning"].level == "top_level"
    assert proposed["Neural Text Classification"].level == "subtopic"
    assert proposed["Neural Text Classification"].parent_names == (
        "Machine Learning",
    )
    assert "Weak Topic" not in proposed
    assert "OpenAI" not in proposed

    existing = topic_taxonomy.assign(
        proposals=[
            {
                "name": "Neural Network Models",
                "parent_names": [],
                "evidence": "We evaluate neural network models.",
                "confidence": 0.8,
            }
        ],
        existing_topics=[
            {
                "name": "Neural Network Model",
                "key": "neural-network-model",
                "level": "top_level",
                "status": "active",
            }
        ],
    )
    assert [item.name for item in existing] == ["Neural Network Model"]
    aliased = topic_taxonomy.assign(
        proposals=[
            {
                "name": "NLP",
                "evidence": "The study evaluates NLP.",
                "confidence": 0.94,
            }
        ],
        existing_topics=[
            {
                "name": "Natural Language Processing",
                "aliases": ["NLP"],
            }
        ],
    )
    assert aliased[0].name == "Natural Language Processing"
    assert aliased[0].aliases == ("NLP",)
    diagnostics = []
    ambiguous = topic_taxonomy.assign(
        proposals=[
            {
                "name": "Machine Learning",
                "evidence": "The report discusses machine learning.",
                "confidence": 0.96,
            }
        ],
        existing_topics=[
            {
                "name": "Machine Learning",
            },
            {"name": "machine learning"},
        ],
        diagnostics=diagnostics,
    )
    assert ambiguous == []
    assert diagnostics[0]["diagnostic_type"] == "ambiguous_topic_resolution"


def test_mentions_and_resolution() -> None:
    extractor = EntityExtractor(client=FakeModelClient())
    result = extractor.extract(
        "Ahmed is the CEO of Example Company.",
        document_title="Example Paper",
        chunk_id="chunk-1",
    )
    assert len(result.entities) == 2
    assert len(result.mentions) == 1
    assert len(result.relations) == 1, "Unsupported relation types must be rejected"

    registry = PaperEntityRegistry("paper-1")
    for entity in result.entities:
        registry.register(entity)
    ahmed_id = registry.resolve(name="Ahmed", category="Person")
    assert ahmed_id
    assert registry.resolve(name="the CEO", category="Person") is None
    assert registry.resolve(entity_id=ahmed_id) == ahmed_id


def test_topic_proposals_require_grounded_evidence() -> None:
    extractor = EntityExtractor(client=FakeModelClient())
    result = extractor._parse_json_response(
        """{
          "topics": [
            {"name": "Speaker Diarization", "parent_names": ["NLP"],
             "aliases": ["Diarization"],
             "evidence": "We evaluate speaker diarization.",
             "confidence": 0.9},
            {"name": "Unsupported Topic", "parent_names": [],
             "evidence": "This phrase is not present.", "confidence": 0.99}
          ]
        }""",
        source_text="We evaluate speaker diarization.",
    )
    assert [topic.name for topic in result.topics] == ["Speaker Diarization"]


def test_quality_filter_and_alias_merge() -> None:
    extractor = EntityExtractor(client=FakeModelClient())
    noisy = """{
      "entities": [
        {"name": "Zhu et al.", "category": "Person",
         "evidence": "Zhu et al. proposed this method.", "confidence": 0.99},
        {"name": "The method achieves accuracy.", "category": "Finding",
         "evidence": "The method achieves accuracy.", "confidence": 0.99},
        {"name": "Page 3", "category": "Page",
         "evidence": "Page 3", "confidence": 0.99},
        {"name": "OpenAI", "category": "Software",
         "evidence": "OpenAI released the software.", "confidence": 0.99}
      ], "mentions": [], "relations": []
    }"""
    result = extractor._parse_json_response(noisy)
    assert [entity.name for entity in result.entities] == ["OpenAI"]

    registry = PaperEntityRegistry("paper-aliases")
    first = registry.register(ExtractedEntity("Feature Matching", "Method"))
    second = registry.register(ExtractedEntity("Feature Matching (FM)", "Method"))
    assert first.entity_id == second.entity_id
    assert "FM" in first.aliases


def test_same_name_is_scoped_conservatively() -> None:
    paper_a = PaperEntityRegistry("paper-a")
    paper_b = PaperEntityRegistry("paper-b")
    a = paper_a.register(ExtractedEntity("Ahmed Hassan", "Person", "CEO of A"))
    b = paper_b.register(ExtractedEntity("Ahmed Hassan", "Person", "Doctor at B"))
    assert a.entity_id != b.entity_id

    same_paper = PaperEntityRegistry("paper-c")
    ceo = same_paper.register(ExtractedEntity("Ahmed Hassan", "Person", "CEO of A"))
    doctor = same_paper.register(ExtractedEntity("Ahmed Hassan", "Person", "Doctor at B"))
    assert ceo.entity_id != doctor.entity_id
    assert same_paper.resolve(name="Ahmed Hassan", category="Person") is None


def test_cloud_client_contract() -> None:
    client = FakeCloudClient(
        api_key="test-key",
        llm_model="provider/chat-model",
        embed_model="provider/embed-model",
    )
    with patch("graph_rag.llm.model_client.cost_tracker.record"):
        assert client.generate("hello") == "cloud response"
        assert client.get_embeddings_batch(["first", "second"]) == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]


def test_batch_ingestion_excludes_metadata_catalog() -> None:
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        metadata = folder / "metadata.csv"
        metadata.write_text("title,local_pdf\n", encoding="utf-8")
        (folder / "records.csv").write_text("name,value\nA,1\n", encoding="utf-8")
        (folder / "paper.pdf").write_bytes(b"%PDF-1.4")

        candidates = IngestionPipeline._batch_input_files(folder, metadata)

        assert [path.name for path in candidates] == ["paper.pdf", "records.csv"]


def test_role_specific_model_routing() -> None:
    clients = {
        id(extraction_model_client),
        id(chat_model_client),
        id(evaluation_model_client),
        id(embedding_model_client),
    }
    assert len(clients) == 4, "Each model responsibility must use a separate client"
    assert extraction_model_client.llm_model == settings.EXTRACTION_MODEL
    assert chat_model_client.llm_model == settings.CHAT_MODEL
    assert evaluation_model_client.llm_model == settings.EVALUATION_MODEL
    assert embedding_model_client.embed_model == settings.EMBEDDING_MODEL
    assert not extraction_model_client.embed_model
    assert not chat_model_client.embed_model
    assert not evaluation_model_client.embed_model
    assert not embedding_model_client.llm_model


def test_document_extraction_cache_reuses_ready_result() -> None:
    class CountingExtractor:
        calls = 0

        def extract_document(self, document):
            self.calls += 1
            return GraphExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="K-GAN",
                        category="Method",
                        evidence="We propose K-GAN.",
                        confidence=0.98,
                    )
                ],
                model_responses=[
                    {
                        "raw_model_output": '{"nodes":[],"relationships":[]}',
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                    }
                ],
            )

    class FakeGraph:
        def get_graph_vocabulary(self):
            return {"entity_types": [], "relationship_types": []}

        def get_known_entity_context(self, candidate_names):
            return []

        def find_entity_candidates(self, **kwargs):
            return []

    class FakeVectorStore:
        @staticmethod
        def search_entity_vectors_batch(queries):
            return [[] for _ in queries]

    fake = CountingExtractor()
    original_extractor = pipeline_module.entity_extractor
    original_cache_dir = settings.EXTRACTION_CACHE_DIR
    original_db_path = settings.SQLITE_DB_PATH
    original_prepare = pipeline_module.prepare_document_input
    original_spacy = pipeline_module.spacy_candidate_extractor
    pipeline = object.__new__(IngestionPipeline)
    pipeline.graph_db = FakeGraph()
    pipeline.vector_db = FakeVectorStore()
    structured = StructuredDocumentInput(
        document_id="paper-hash",
        title="K-GAN Paper",
        document_type="research_paper",
        language="en",
        blocks=[
            SourceBlock(
                source_ref="page:1",
                block_type="paragraph",
                text="We propose K-GAN.",
            )
        ],
    )
    fake_spacy = SimpleNamespace(
        extract=lambda *args: {"entities": [], "noun_phrases": []},
        token_count=lambda text, *args: len(text.split()),
    )
    parsed_document = SimpleNamespace(language="en")
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            temp_path = Path(tmp)
            object.__setattr__(settings, "EXTRACTION_CACHE_DIR", temp_path / "cache")
            object.__setattr__(settings, "SQLITE_DB_PATH", temp_path / "test.sqlite")
            database.init_db()
            pipeline_module.entity_extractor = fake
            pipeline_module.prepare_document_input = lambda _: structured
            pipeline_module.spacy_candidate_extractor = fake_spacy
            first = pipeline._extract_document_graph(
                parsed_document,
                "K-GAN Paper",
                "paper-hash",
                file_hash="paper-hash",
            )
            second = pipeline._extract_document_graph(
                parsed_document,
                "K-GAN Paper",
                "paper-hash",
                file_hash="paper-hash",
                cache_only=True,
            )
            assert fake.calls == 1, "A graph rebuild must reuse the ready extraction cache"
            assert first.cache_stage == second.cache_stage == "ready"
            assert first.entities[0].entity_id == second.entities[0].entity_id
            assert second.model_responses == first.model_responses
    finally:
        pipeline_module.entity_extractor = original_extractor
        pipeline_module.prepare_document_input = original_prepare
        pipeline_module.spacy_candidate_extractor = original_spacy
        object.__setattr__(settings, "EXTRACTION_CACHE_DIR", original_cache_dir)
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_extraction_cache_signature_tracks_language_and_context_settings() -> None:
    english_signature = IngestionPipeline._cache_signature("en")
    arabic_signature = IngestionPipeline._cache_signature("ar")
    assert english_signature != arabic_signature
    assert english_signature["cache_format_version"] == "2"
    assert "evidence_match_version" in english_signature
    assert "context_tokens" in english_signature
    assert "spacy_models" in english_signature


def test_failed_model_response_is_cached_and_not_retried_implicitly() -> None:
    from graph_rag.graph.entity_extractor import GraphExtractionError

    class FailingExtractor:
        calls = 0

        def extract_document(self, document):
            self.calls += 1
            raise GraphExtractionError(
                "mock truncated graph response",
                model_responses=[
                    {
                        "raw_model_output": '{"nodes":',
                        "finish_reason": "length",
                        "truncated": True,
                        "source_refs": ["page:1"],
                    }
                ],
            )

    class FakeGraph:
        def get_graph_vocabulary(self):
            return {"entity_types": [], "relationship_types": []}

        def get_known_entity_context(self, candidate_names):
            return []

    fake = FailingExtractor()
    original_extractor = pipeline_module.entity_extractor
    original_cache_dir = settings.EXTRACTION_CACHE_DIR
    original_db_path = settings.SQLITE_DB_PATH
    original_prepare = pipeline_module.prepare_document_input
    original_spacy = pipeline_module.spacy_candidate_extractor
    pipeline = object.__new__(IngestionPipeline)
    pipeline.graph_db = FakeGraph()
    pipeline.vector_db = object()
    structured = StructuredDocumentInput(
        document_id="failed-doc",
        title="Failed document",
        document_type="report",
        language="en",
        blocks=[
            SourceBlock(
                source_ref="page:1",
                block_type="paragraph",
                text="A source sentence.",
            )
        ],
    )
    fake_spacy = SimpleNamespace(
        extract=lambda *args: {"entities": [], "noun_phrases": []},
        token_count=lambda text, *args: len(text.split()),
    )
    parsed_document = SimpleNamespace(language="en")

    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            temp_path = Path(tmp)
            object.__setattr__(settings, "EXTRACTION_CACHE_DIR", temp_path / "cache")
            object.__setattr__(settings, "SQLITE_DB_PATH", temp_path / "test.sqlite")
            database.init_db()
            pipeline_module.entity_extractor = fake
            pipeline_module.prepare_document_input = lambda _: structured
            pipeline_module.spacy_candidate_extractor = fake_spacy

            try:
                pipeline._extract_document_graph(
                    parsed_document,
                    "Failed document",
                    "failed-doc",
                    file_hash="failed-doc-hash",
                )
            except RuntimeError as exc:
                assert "cached for diagnosis" in str(exc)
            else:
                raise AssertionError("A failed segment must fail document extraction")

            cache_path = pipeline._cache_path("failed-doc")
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            assert payload["result"]["cache_stage"] == "failed"
            assert payload["result"]["model_responses"][0]["truncated"] is True

            try:
                pipeline._extract_document_graph(
                    parsed_document,
                    "Failed document",
                    "failed-doc",
                    file_hash="failed-doc-hash",
                )
            except RuntimeError as exc:
                assert "--refresh-extraction" in str(exc)
            else:
                raise AssertionError("Failed cached extraction must require explicit refresh")
            assert fake.calls == 1
    finally:
        pipeline_module.entity_extractor = original_extractor
        pipeline_module.prepare_document_input = original_prepare
        pipeline_module.spacy_candidate_extractor = original_spacy
        object.__setattr__(settings, "EXTRACTION_CACHE_DIR", original_cache_dir)
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_failed_segment_is_split_without_repeating_successful_segments() -> None:
    from graph_rag.graph.entity_extractor import GraphExtractionError

    class FakeExtractionChain:
        def __init__(self):
            self.source_ref_calls = []

        def batch(self, segments, config, return_exceptions):
            assert return_exceptions is True
            results = []
            for segment in segments:
                source_refs = tuple(block.source_ref for block in segment.blocks)
                self.source_ref_calls.append(source_refs)
                if source_refs == ("bad",):
                    results.append(
                        GraphExtractionError(
                            "mock malformed JSON",
                            model_responses=[{"raw_model_output": '{"nodes":'}],
                        )
                    )
                    continue
                results.append(
                    GraphExtractionResult(
                        entities=[
                            ExtractedEntity(
                                name=source_refs[0],
                                category="Test Entity",
                            )
                        ]
                    )
                )
            return results

    good = StructuredDocumentInput(
        document_id="retry-doc",
        title="Retry document",
        document_type="report",
        language="en",
        blocks=[SourceBlock("good", "paragraph", "A successful source region.")],
    )
    bad_text = " ".join(
        f"Sentence {index} contains grounded extraction evidence."
        for index in range(500)
    )
    bad = StructuredDocumentInput(
        document_id="retry-doc",
        title="Retry document",
        document_type="report",
        language="en",
        blocks=[SourceBlock("bad", "paragraph", bad_text)],
    )
    chain = FakeExtractionChain()
    pipeline = object.__new__(IngestionPipeline)

    successes, terminal_errors, attempted_errors = (
        pipeline._extract_segments_with_retry(
            chain,
            [good, bad],
            document_id="retry-doc",
            document_name="retry.pdf",
        )
    )

    assert not terminal_errors
    assert len(attempted_errors) == 1
    assert chain.source_ref_calls.count(("good",)) == 1
    assert chain.source_ref_calls.count(("bad",)) == 1
    assert len(successes) > 2
    assert any(
        source_refs[0].startswith("bad:")
        for source_refs in chain.source_ref_calls
        if source_refs
    )


def test_retry_circuit_opens_when_most_of_a_batch_fails() -> None:
    from graph_rag.graph.entity_extractor import GraphExtractionError

    class AlwaysFailingChain:
        calls = 0

        def batch(self, segments, config, return_exceptions):
            self.calls += 1
            return [
                GraphExtractionError(f"invalid JSON for segment {index}")
                for index, _ in enumerate(segments)
            ]

    segments = [
        StructuredDocumentInput(
            document_id="circuit-doc",
            title="Circuit document",
            document_type="report",
            language="en",
            blocks=[
                SourceBlock(
                    f"segment:{index}",
                    "paragraph",
                    "A long failing sentence. " * 500,
                )
            ],
        )
        for index in range(3)
    ]
    chain = AlwaysFailingChain()
    pipeline = object.__new__(IngestionPipeline)

    successes, terminal_errors, attempted_errors = (
        pipeline._extract_segments_with_retry(
            chain,
            segments,
            document_id="circuit-doc",
            document_name="circuit.pdf",
        )
    )

    assert not successes
    assert len(terminal_errors) == 3
    assert len(attempted_errors) == 3
    assert chain.calls == 1, "Circuit breaker must prevent recursive paid retries"


def test_incompatible_cache_requires_explicit_refresh_but_legacy_cache_is_readable() -> None:
    pipeline = object.__new__(IngestionPipeline)
    original_cache_dir = settings.EXTRACTION_CACHE_DIR
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            object.__setattr__(settings, "EXTRACTION_CACHE_DIR", Path(tmp) / "cache")
            path = pipeline._cache_path("cached-document")
            path.parent.mkdir(parents=True, exist_ok=True)
            empty_result = {
                "entities": [],
                "mentions": [],
                "relations": [],
                "topics": [],
                "diagnostics": [],
                "document_type": "report",
                "cache_stage": "ready",
            }
            payload = {
                "document_id": "cached-document",
                "file_hash": "same-content",
                "cache_signature": {
                    **pipeline._cache_signature("en"),
                    "language": "ar",
                },
                "result": empty_result,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                pipeline._load_extraction_cache(
                    "cached-document",
                    "same-content",
                    language="en",
                )
            except RuntimeError as exc:
                assert "--refresh-extraction" in str(exc)
                assert "may incur model charges" in str(exc)
            else:
                raise AssertionError("Incompatible cache must not trigger implicit extraction")

            payload["cache_signature"] = pipeline._legacy_cache_signature()
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = pipeline._load_extraction_cache(
                "cached-document",
                "same-content",
                language="en",
            )
            assert loaded is not None
            assert loaded.model_responses == []
            assert loaded.cache_metadata["legacy_cache"] is True

            payload["cache_signature"] = pipeline._cache_signature("en")
            payload["result"] = {
                **empty_result,
                "cache_stage": "failed",
                "model_responses": [
                    {
                        "raw_model_output": '{"nodes":',
                        "finish_reason": "length",
                        "truncated": True,
                    }
                ],
                "cache_metadata": {
                    "raw_model_responses_available": True,
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            failed = pipeline._load_extraction_cache(
                "cached-document",
                "same-content",
                language="en",
            )
            assert failed is not None
            assert failed.model_responses[0]["truncated"] is True
            try:
                pipeline._raise_for_failed_cache(failed, "cached-document")
            except RuntimeError as exc:
                assert "--refresh-extraction" in str(exc)
            else:
                raise AssertionError("Failed cache must not trigger implicit extraction")
    finally:
        object.__setattr__(settings, "EXTRACTION_CACHE_DIR", original_cache_dir)


def test_entity_resolver_uses_exact_candidates_and_rejects_ambiguity() -> None:
    class FakeGraph:
        def find_entity_candidates(self, **kwargs):
            if kwargs["name"] == "Alex":
                return [
                    {
                        "entity_id": "person:alex",
                        "name": "Alex Smith",
                        "category": "Person",
                        "aliases": ["Alex"],
                        "attributes_json": '{"organization": "Acme"}',
                    }
                ]
            if kwargs["name"] == "Sam Lee":
                return [
                    {"entity_id": "person:one", "name": "Sam Lee", "category": "Person"},
                    {"entity_id": "person:two", "name": "Sam Lee", "category": "Person"},
                ]
            return []

    class FakeVectorStore:
        @staticmethod
        def search_entity_vectors_batch(queries):
            return [
                [
                    {
                        "entity_id": "person:likely",
                        "name": "Likely Existing Person",
                        "category": "Person",
                        "similarity_score": 0.96,
                    }
                ]
                if "Likely" in query
                else [
                    {
                        "entity_id": "person:ambiguous-one",
                        "name": "Ambiguous Person One",
                        "category": "Person",
                        "similarity_score": 0.95,
                    },
                    {
                        "entity_id": "person:ambiguous-two",
                        "name": "Ambiguous Person Two",
                        "category": "Person",
                        "similarity_score": 0.94,
                    },
                ]
                for query in queries
            ]

    entities = [
        ExtractedEntity(
            name="Alex",
            category="Person",
            entity_id="local:alex",
            attributes={"organization": "Acme"},
        ),
        ExtractedEntity(
            name="Sam Lee",
            category="Person",
            entity_id="local:sam",
        ),
        ExtractedEntity(
            name="Likely Person",
            category="Person",
            entity_id="local:likely",
        ),
        ExtractedEntity(
            name="Ambiguous Person",
            category="Person",
            entity_id="local:ambiguous",
        ),
    ]
    resolved, diagnostics, ambiguous, id_map = EntityResolver(
        FakeGraph(),
        FakeVectorStore(),
    ).resolve(entities)

    assert {entity.entity_id for entity in resolved} == {
        "person:alex",
        "person:likely",
    }
    assert "Alex" in next(entity for entity in resolved if entity.entity_id == "person:alex").aliases
    assert id_map == {
        "local:alex": "person:alex",
        "local:sam": "",
        "local:likely": "person:likely",
        "local:ambiguous": "",
    }
    assert ambiguous == {"local:sam", "local:ambiguous"}
    assert len(diagnostics) == 2
    assert all(
        diagnostic["diagnostic_type"] == "ambiguous_entity_resolution"
        for diagnostic in diagnostics
    )


def test_type_normalization_keeps_unseen_types_open() -> None:
    result = GraphExtractionResult(
        entities=[
            ExtractedEntity(name="OpenAI", category="Software", evidence="OpenAI")
        ],
    )
    normalized = TypeNormalizer(None).normalize_result(
        result,
        existing_entity_types=[],
        existing_relationship_types=[],
    )
    assert normalized.entities[0].category == "Software"


def test_entity_vector_upsert_avoids_reembedding_unchanged_entities() -> None:
    class FakeCollection:
        def __init__(self):
            self.rows = {}

        def get(self, ids=None, include=None):
            selected = ids or list(self.rows)
            rows = [self.rows[entity_id] for entity_id in selected if entity_id in self.rows]
            return {
                "ids": [row["id"] for row in rows],
                "documents": [row["document"] for row in rows],
                "metadatas": [row["metadata"] for row in rows],
            }

        def upsert(self, ids, embeddings, documents, metadatas):
            for entity_id, document, metadata in zip(ids, documents, metadatas):
                self.rows[entity_id] = {
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                }

        def update(self, ids, metadatas):
            for entity_id, metadata in zip(ids, metadatas):
                self.rows[entity_id]["metadata"] = metadata

        def delete(self, ids):
            for entity_id in ids:
                self.rows.pop(entity_id, None)

    store = object.__new__(VectorStore)
    collection = FakeCollection()
    store._entity_collection = collection
    entity = ExtractedEntity(
        name="Acme",
        category="Organization",
        entity_id="entity:acme",
        aliases=["Acme Corp"],
    )
    with patch(
        "graph_rag.embeddings.vector_store.embedding_model_client.get_embeddings_batch",
        return_value=[[0.1, 0.2]],
    ) as embed:
        assert store.upsert_entity_vectors([entity], "document-a") == 1
        assert store.upsert_entity_vectors([entity], "document-a") == 0
        assert store.upsert_entity_vectors([entity], "document-b") == 0
        assert embed.call_count == 1
    assert collection.rows["entity:acme"]["metadata"]["source_document_ids"] == (
        '["document-a", "document-b"]'
    )
    store.delete_document_entity_vectors("document-a")
    assert "entity:acme" in collection.rows
    store.delete_document_entity_vectors("document-b")
    assert "entity:acme" not in collection.rows


def test_unresolved_reference_persistence() -> None:
    original_db_path = settings.SQLITE_DB_PATH
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            object.__setattr__(settings, "SQLITE_DB_PATH", Path(tmp) / "test.sqlite")
            database.init_db()
            unresolved = ExtractedMention(
                text="the manager",
                mention_id="mention-1",
                mention_type="role",
                attributes={"role": "Manager", "organization": "Acme"},
                evidence="The manager reviewed the work.",
                confidence=0.4,
            )
            resolved = ExtractedMention(
                text="Alex",
                entity_id="person:alex",
                mention_id="mention-2",
                evidence="Alex reviewed the work.",
            )
            database.save_unresolved_references(
                "paper.pdf", "paper-hash", [unresolved, resolved]
            )
            with database.get_db_connection() as conn:
                rows = conn.execute(
                    "SELECT mention_text, evidence FROM unresolved_references"
                ).fetchall()
            assert [(row["mention_text"], row["evidence"]) for row in rows] == [
                ("the manager", "The manager reviewed the work.")
            ]
            database.delete_unresolved_references("paper.pdf")
            with database.get_db_connection() as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM unresolved_references"
                ).fetchone()[0] == 0
    finally:
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_graph_evidence_persists_resolved_mentions() -> None:
    original_db_path = settings.SQLITE_DB_PATH
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            object.__setattr__(settings, "SQLITE_DB_PATH", Path(tmp) / "evidence.sqlite")
            database.init_db()
            mention = ExtractedMention(
                text="the director",
                mention_id="mention-1",
                entity_id="person:director",
                evidence="The director led Acme.",
                confidence=0.93,
                provenance={
                    "page_number": 2,
                    "section": "Results",
                    "source_offsets": [{"start": 10, "end": 33}],
                },
            )
            database.save_graph_evidence(
                "document-1",
                "report.pdf",
                "document-hash",
                GraphExtractionResult(mentions=[mention]),
            )
            with database.get_db_connection() as conn:
                row = conn.execute(
                    """
                    SELECT object_type, object_id, mention_text, evidence_text,
                           page_number, section, source_offsets_json, confidence
                    FROM graph_evidence
                    """
                ).fetchone()
            assert row["object_type"] == "mention"
            assert row["object_id"] == "mention-1"
            assert row["mention_text"] == "the director"
            assert row["evidence_text"] == "The director led Acme."
            assert row["page_number"] == 2
            assert row["section"] == "Results"
            assert json.loads(row["source_offsets_json"]) == [
                {"start": 10, "end": 33}
            ]
            assert row["confidence"] == 0.93
    finally:
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_ingestion_run_journal_resumes_and_reconciles() -> None:
    original_db_path = settings.SQLITE_DB_PATH
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            object.__setattr__(settings, "SQLITE_DB_PATH", Path(tmp) / "journal.sqlite")
            database.init_db()
            upload_id = database.start_ingestion_run(
                "recovery.pdf", "recovery-hash", "upload"
            )
            database.update_ingestion_run(upload_id, "vectors_ready")
            database.finish_ingestion_run(
                upload_id,
                error_message="Simulated interruption",
            )
            assert database.has_incomplete_ingestion_run(
                "recovery.pdf", "recovery-hash"
            )
            assert database.get_incomplete_ingestion_run(
                "recovery-hash", "upload"
            )["document_name"] == "recovery.pdf"

            resumed_id = database.start_ingestion_run(
                "recovery.pdf", "recovery-hash", "upload"
            )
            assert resumed_id == upload_id
            with database.get_db_connection() as conn:
                attempt_count = conn.execute(
                    "SELECT attempt_count FROM ingestion_runs WHERE id = ?",
                    (upload_id,),
                ).fetchone()[0]
            assert attempt_count == 2
            database.finish_ingestion_run(
                resumed_id,
                error_message="Retry also interrupted",
            )

            batch_id = database.start_ingestion_run(
                "recovery.pdf", "recovery-hash", "batch"
            )
            database.update_ingestion_run(batch_id, "graph_written")
            database.finish_ingestion_run(batch_id)
            assert not database.has_incomplete_ingestion_run(
                "recovery.pdf", "recovery-hash"
            )
            assert database.get_incomplete_ingestion_run(
                "recovery-hash", "upload"
            ) is None
            with database.get_db_connection() as conn:
                statuses = [
                    row[0]
                    for row in conn.execute(
                        "SELECT status FROM ingestion_runs ORDER BY id"
                    ).fetchall()
                ]
            assert statuses == ["completed", "completed"]
    finally:
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_failed_upload_rolls_back_and_retries_without_external_services() -> None:
    class FakeMetadata:
        rows = {}

        def find_by_filename(self, filename):
            return self.rows.get(filename)

        def append_document(self, row):
            self.rows[row["local_pdf"]] = row

        def mark_as_ingested(self, **kwargs):
            self.rows[kwargs["local_pdf"]].update(kwargs)

        def remove_document(self, filename):
            return int(self.rows.pop(filename, None) is not None)

    class FakeDedup:
        def check_bytes(self, **kwargs):
            return SimpleNamespace(is_duplicate=False)

    class FakeVectorStore:
        def __init__(self):
            self.deleted = []

        def add_chunks(self, chunks):
            return None

        def delete_document_chunks(self, filename):
            self.deleted.append(filename)

        def upsert_entity_vectors(self, entities, document_id):
            return len(entities)

        def delete_document_entity_vectors(self, document_id):
            return None

    class FakeGraph:
        def __init__(self):
            self.fail_entity_write = True
            self.deleted = []

        def add_document_node(self, **kwargs):
            return True

        def add_entities_and_relations(self, *args, **kwargs):
            if self.fail_entity_write:
                self.fail_entity_write = False
                return False
            return True

        def delete_document(self, filename, document_id=""):
            self.deleted.append(filename)
            return True

    class FakeFileHandler:
        def is_supported(self, filename):
            return True

        def parse(self, *args, **kwargs):
            return SimpleNamespace(
                title="Retry paper",
                file_type="txt",
                mime_type="text/plain",
                language="en",
                table_data=[],
                pages=[],
                metadata={},
                document_id="",
                document_type="unknown",
            )

    class FakeChunker:
        def chunk_document(self, parsed_doc):
            return [SimpleNamespace(text="paper text")]

    original_pdf_dir = settings.PDF_DIR
    original_db_path = settings.SQLITE_DB_PATH
    original_file_handler = pipeline_module.file_handler
    original_chunker = pipeline_module.chunker
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            root = Path(tmp)
            object.__setattr__(settings, "PDF_DIR", root / "pdfs")
            object.__setattr__(settings, "SQLITE_DB_PATH", root / "journal.sqlite")
            database.init_db()
            pipeline_module.file_handler = FakeFileHandler()
            pipeline_module.chunker = FakeChunker()

            pipeline = object.__new__(IngestionPipeline)
            pipeline.meta_mgr = FakeMetadata()
            pipeline.dedup = FakeDedup()
            pipeline.vector_db = FakeVectorStore()
            pipeline.graph_db = FakeGraph()
            pipeline._extract_document_graph = lambda *args, **kwargs: GraphExtractionResult()

            first = pipeline.ingest_uploaded_file(b"retry me", "retry.txt")
            assert first.status == "failed"
            assert pipeline.graph_db.deleted == ["retry.txt"]
            assert pipeline.vector_db.deleted == ["retry.txt"]
            assert not (settings.PDF_DIR / "retry.txt").exists()
            assert database.get_incomplete_ingestion_run(
                first.file_hash, "upload"
            )["stage"] == "failed"

            second = pipeline.ingest_uploaded_file(b"retry me", "retry.txt")
            assert second.status == "completed"
            assert second.filename == "retry.txt"
            assert (settings.PDF_DIR / "retry.txt").exists()
            assert not database.has_incomplete_ingestion_run(
                "retry.txt", second.file_hash
            )
            with database.get_db_connection() as conn:
                run = conn.execute(
                    "SELECT status, attempt_count FROM ingestion_runs WHERE file_hash = ?",
                    (second.file_hash,),
                ).fetchone()
            assert (run["status"], run["attempt_count"]) == ("completed", 2)
    finally:
        pipeline_module.file_handler = original_file_handler
        pipeline_module.chunker = original_chunker
        object.__setattr__(settings, "PDF_DIR", original_pdf_dir)
        object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)


def test_metadata_csv_keeps_previous_file_if_replacement_fails() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
        metadata_path = Path(tmp) / "metadata.csv"
        manager = MetadataManager(metadata_path)
        manager.append_document(
            {
                "local_pdf": "stable.txt",
                "title": "Stable document",
                "file_hash": "stable-hash",
            }
        )
        previous_contents = metadata_path.read_bytes()
        with patch(
            "pandas.DataFrame.to_csv",
            side_effect=OSError("simulated write failure"),
        ):
            try:
                manager.append_document(
                    {
                        "local_pdf": "new.txt",
                        "title": "New document",
                        "file_hash": "new-hash",
                    }
                )
            except OSError:
                pass
            else:
                raise AssertionError("Metadata write failure should be surfaced")
        assert metadata_path.read_bytes() == previous_contents
        assert not list(Path(tmp).glob("*.tmp"))
        assert not list(Path(tmp).glob(".*.tmp"))


def test_answer_prompt_follows_question_language() -> None:
    question = "ما هي الفكرة الرئيسية؟"
    system_prompt, user_prompt = build_prompts(
        question=question,
        context_text="مقتطف بحثي",
        classification=None,
    )
    assert "same language as the user's question" in system_prompt
    assert "unless they explicitly request another language" in system_prompt
    assert question in user_prompt


def test_graph_rebuild_repairs_missing_vectors() -> None:
    assert IngestionPipeline._needs_vector_write(False, 0)
    assert IngestionPipeline._needs_vector_write(False, 5)
    assert IngestionPipeline._needs_vector_write(True, 0)
    assert not IngestionPipeline._needs_vector_write(True, 5)
    assert IngestionPipeline._needs_vector_replace(True, False, 5)
    assert IngestionPipeline._needs_vector_replace(False, True, 5)
    assert not IngestionPipeline._needs_vector_replace(False, False, 5)
    assert not IngestionPipeline._needs_vector_replace(True, False, 0)


def test_dynamic_entity_cypher_is_renderable() -> None:
    """Execute the dynamic label path so f-string Cypher maps cannot regress."""

    class FakeResult:
        def consume(self):
            return self

        def single(self):
            return None

    class FakeSession:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            self.queries.append(query)
            return FakeResult()

    class FakeDriver:
        def __init__(self):
            self.fake_session = FakeSession()

        def session(self):
            return self.fake_session

        def close(self):
            return None

    builder = GraphBuilder()
    driver = FakeDriver()
    builder._driver = driver
    result = GraphExtractionResult(
        entities=[
            ExtractedEntity(
                name="Ahmed Hassan",
                category="Person",
                entity_id="person:ahmed-hassan",
            )
        ]
    )
    assert builder.add_entities_and_relations("paper.pdf", result)
    assert any(
        "MERGE (e:Entity {uid: $uid})" in query
        for query in driver.fake_session.queries
    )


def test_compact_graph_write_policy() -> None:
    """Byline metadata and mention edges must not inflate the Neo4j graph."""

    class FakeResult:
        def consume(self):
            return self

        def single(self):
            return None

    class FakeSession:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            self.queries.append(query)
            return FakeResult()

    class FakeDriver:
        def __init__(self):
            self.fake_session = FakeSession()

        def session(self):
            return self.fake_session

        def close(self):
            return None

    builder = GraphBuilder()
    driver = FakeDriver()
    builder._driver = driver
    result = GraphExtractionResult(
        entities=[
            ExtractedEntity(
                name="Paper Author",
                category="Person",
                description="Author of the paper",
                entity_id="person:paper-author",
            ),
            ExtractedEntity(
                name="Author University",
                category="Organization",
                description="Affiliation of Paper Author",
                entity_id="organization:author-university",
            ),
            ExtractedEntity(
                name="Useful Method",
                category="Method",
                entity_id="method:useful-method",
            ),
        ],
        mentions=[SimpleNamespace(
            entity_id="method:useful-method",
            chunk_id="paper:0",
            text="Useful Method",
            evidence="We use Useful Method.",
        )],
    )
    assert builder.add_entities_and_relations("paper.pdf", result)
    rendered = "\n".join(driver.fake_session.queries)
    assert "MENTIONS" not in rendered
    assert "Paper Author" not in rendered
    assert rendered.count("CONTAINS_ENTITY") == 0
    assert "USES_METHOD" not in rendered


def test_graph_builder_writes_proposed_topic_parents() -> None:
    class FakeResult:
        def __init__(self, rows=()):
            self.rows = rows

        def __iter__(self):
            return iter(self.rows)

        def consume(self):
            return self

        def single(self):
            return None

    class FakeSession:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            self.calls.append((query, params))
            return FakeResult()

    class FakeDriver:
        def __init__(self):
            self.fake_session = FakeSession()

        def session(self):
            return self.fake_session

        def close(self):
            return None

    builder = GraphBuilder()
    driver = FakeDriver()
    builder._driver = driver
    assert builder.add_document_node(
        "topics.pdf",
        "Topic proposals",
        metadata={
            "topic_proposals": [
                {
                    "name": "Speaker Diarization",
                    "parent_names": ["Natural Language Processing"],
                    "aliases": ["Diarization"],
                    "evidence": "We evaluate speaker diarization.",
                    "confidence": 0.94,
                }
            ]
        },
    )
    subtopic_write = next(
        params
        for query, params in driver.fake_session.calls
        if "UNWIND $parent_uids AS parent_uid" in query
    )
    assert subtopic_write["parent_uids"] == [builder._uid(
        "topic:natural-language-processing"
    )]
    assert subtopic_write["document_id"] == "topics.pdf"
    assert subtopic_write["evidence"] == "We evaluate speaker diarization."
    assert subtopic_write["aliases"] == ["Diarization"]
    assert builder.add_document_node(
        "new-topic.pdf",
        "A new research area",
        metadata={
            "topic_proposals": [
                {
                    "name": "Computational Linguistics",
                    "parent_names": [],
                    "evidence": "We study computational linguistics.",
                    "confidence": 0.88,
                }
            ]
        },
    )
    top_level_write = next(
        params
        for query, params in driver.fake_session.calls
        if "MERGE (t)-[r:HAS_DOCUMENT]" in query
        and params.get("document_uid") == builder._uid("new-topic.pdf")
    )
    assert top_level_write["evidence"] == "We study computational linguistics."


def test_role_lookup_rejects_ambiguous_matches() -> None:
    class FakeSession:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, *args, **kwargs):
            return iter(self.rows)

    class FakeDriver:
        def __init__(self, rows):
            self.rows = rows

        def session(self):
            return FakeSession(self.rows)

    builder = GraphBuilder()
    builder._driver = FakeDriver([
        {"entity_id": "person:one", "name": "First"},
        {"entity_id": "person:two", "name": "Second"},
    ])
    assert builder.find_person_by_role("CEO", "Example Co.") is None

    builder._driver = FakeDriver([
        {"entity_id": "person:one", "name": "First"},
    ])
    assert builder.find_person_by_role("CEO", "Example Co.") == {
        "entity_id": "person:one",
        "name": "First",
    }


if __name__ == "__main__":
    test_topics()
    test_mentions_and_resolution()
    test_topic_proposals_require_grounded_evidence()
    test_quality_filter_and_alias_merge()
    test_same_name_is_scoped_conservatively()
    test_cloud_client_contract()
    test_batch_ingestion_excludes_metadata_catalog()
    test_role_specific_model_routing()
    test_document_extraction_cache_reuses_ready_result()
    test_extraction_cache_signature_tracks_language_and_context_settings()
    test_failed_model_response_is_cached_and_not_retried_implicitly()
    test_failed_segment_is_split_without_repeating_successful_segments()
    test_retry_circuit_opens_when_most_of_a_batch_fails()
    test_incompatible_cache_requires_explicit_refresh_but_legacy_cache_is_readable()
    test_entity_resolver_uses_exact_candidates_and_rejects_ambiguity()
    test_type_normalization_keeps_unseen_types_open()
    test_entity_vector_upsert_avoids_reembedding_unchanged_entities()
    test_unresolved_reference_persistence()
    test_graph_evidence_persists_resolved_mentions()
    test_ingestion_run_journal_resumes_and_reconciles()
    test_failed_upload_rolls_back_and_retries_without_external_services()
    test_metadata_csv_keeps_previous_file_if_replacement_fails()
    test_answer_prompt_follows_question_language()
    test_graph_rebuild_repairs_missing_vectors()
    test_dynamic_entity_cypher_is_renderable()
    test_compact_graph_write_policy()
    test_graph_builder_writes_proposed_topic_parents()
    test_role_lookup_rejects_ambiguous_matches()
    print("Graph redesign tests passed")

