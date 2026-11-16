"""Fast unit checks for the graph-redesign foundations.

These checks do not call OpenRouter, ChromaDB, or Neo4j.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.graph.entity_extractor import EntityExtractor, ExtractedEntity
from graph_rag.graph.entity_resolution import PaperEntityRegistry
from graph_rag.graph.topic_taxonomy import topic_taxonomy
from graph_rag.config import settings
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
    assignments = topic_taxonomy.assign("NLP; Medical Informatics")
    by_name = {item.name: item for item in assignments}
    assert by_name["Natural Language Processing"].status == "active"
    assert by_name["Medical Informatics"].status == "candidate"


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


if __name__ == "__main__":
    test_topics()
    test_mentions_and_resolution()
    test_same_name_is_scoped_conservatively()
    test_cloud_client_contract()
    test_role_specific_model_routing()
    print("Graph redesign tests passed")

