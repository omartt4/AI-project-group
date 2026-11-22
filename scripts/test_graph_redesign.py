"""Fast unit checks for the graph-redesign foundations.

These checks do not call OpenRouter, ChromaDB, or Neo4j.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.graph.entity_extractor import EntityExtractor, ExtractedEntity, GraphExtractionResult
from graph_rag.graph.entity_resolution import PaperEntityRegistry
from graph_rag.graph.graph_builder import GraphBuilder
from graph_rag.graph.topic_taxonomy import topic_taxonomy
from graph_rag.config import settings
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
             "attributes": {"role": "CEO", "organization": "Example Company"}},
            {"entity_id": "", "name": "Example Company", "category": "Organization",
             "description": "An example company", "aliases": [], "attributes": {}}
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
    def _post(self, path, payload):
        if path == "/chat/completions":
            return {"choices": [{"message": {"content": "cloud response"}}]}
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


def test_topics() -> None:
    assignments = topic_taxonomy.assign(
        "NLP; Medical Informatics; Speaker Diarization"
    )
    by_name = {item.name: item for item in assignments}
    assert by_name["Natural Language Processing"].status == "active"
    assert by_name["Natural Language Processing"].level == "top_level"
    assert by_name["Medical Informatics"].status == "candidate"
    assert by_name["Medical Informatics"].level == "subtopic"
    assert by_name["Medical Informatics"].parent_names == (
        "Natural Language Processing",
    )
    assert by_name["Speaker Diarization"].level == "subtopic"
    assert by_name["Speaker Diarization"].parent_names == (
        "Natural Language Processing",
    )

    vision = topic_taxonomy.assign("Computer Vision; Generative Models")
    vision_by_name = {item.name: item for item in vision}
    assert vision_by_name["Generative Models"].parent_names == (
        "Computer Vision",
    )


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


def test_quality_filter_and_alias_merge() -> None:
    extractor = EntityExtractor(client=FakeModelClient())
    noisy = """{
      "entities": [
        {"name": "Zhu et al.", "category": "Person"},
        {"name": "Table 3 K values", "category": "Metric"},
        {"name": "Accuracy", "category": "Metric"},
        {"name": "Spiders", "category": "Topic"},
        {"name": "Diarization error rate (DER)", "category": "Metric"},
        {"name": "K-GAN", "category": "Method"}
      ], "mentions": [], "relations": []
    }"""
    result = extractor._parse_json_response(noisy)
    assert [entity.name for entity in result.entities] == ["K-GAN"]

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
    assert client.generate("hello") == "cloud response"
    assert client.get_embeddings_batch(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


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


def test_paper_level_extraction_cache() -> None:
    class CountingExtractor:
        calls = 0

        def extract(self, *args, **kwargs):
            self.calls += 1
            return GraphExtractionResult(
                entities=[ExtractedEntity(name="K-GAN", category="Method")]
            )

    fake = CountingExtractor()
    original_extractor = pipeline_module.entity_extractor
    original_cache_dir = settings.EXTRACTION_CACHE_DIR
    pipeline = object.__new__(IngestionPipeline)
    pipeline.graph_db = SimpleNamespace(find_person_by_role=lambda *args: None)
    chunks = [
        SimpleNamespace(
            text="Abstract. We propose the K-GAN method and evaluate the approach.",
            chunk_index=0,
            page_number=1,
        ),
        SimpleNamespace(
            text="References\n[1] unrelated cited work",
            chunk_index=1,
            page_number=2,
        ),
    ]
    try:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp:
            object.__setattr__(settings, "EXTRACTION_CACHE_DIR", Path(tmp))
            pipeline_module.entity_extractor = fake
            first = pipeline._extract_paper_graph(
                chunks, "K-GAN Paper", "paper.pdf", file_hash="hash-1"
            )
            second = pipeline._extract_paper_graph(
                chunks, "K-GAN Paper", "paper.pdf", file_hash="hash-1"
            )
            assert fake.calls == 1, "A rebuild must reuse the paid extraction cache"
            assert first.entities[0].entity_id == second.entities[0].entity_id
    finally:
        pipeline_module.entity_extractor = original_extractor
        object.__setattr__(settings, "EXTRACTION_CACHE_DIR", original_cache_dir)


def test_graph_rebuild_repairs_missing_vectors() -> None:
    assert IngestionPipeline._needs_vector_write(False, 0)
    assert IngestionPipeline._needs_vector_write(False, 5)
    assert IngestionPipeline._needs_vector_write(True, 0)
    assert not IngestionPipeline._needs_vector_write(True, 5)


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
    assert "USES_METHOD" in rendered


if __name__ == "__main__":
    test_topics()
    test_mentions_and_resolution()
    test_quality_filter_and_alias_merge()
    test_same_name_is_scoped_conservatively()
    test_cloud_client_contract()
    test_role_specific_model_routing()
    test_paper_level_extraction_cache()
    test_graph_rebuild_repairs_missing_vectors()
    test_dynamic_entity_cypher_is_renderable()
    test_compact_graph_write_policy()
    print("Graph redesign tests passed")

