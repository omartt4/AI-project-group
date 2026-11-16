"""Neo4j writer for the evidence-first graph schema."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from graph_rag.config import settings
from graph_rag.graph.entity_extractor import ALLOWED_RELATION_TYPES, GraphExtractionResult
from graph_rag.graph.topic_taxonomy import topic_taxonomy

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build Paper, Topic, Chunk, Mention, and Entity nodes with provenance."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.dataset_id = settings.DATASET_ID
        self._driver: Optional[Driver] = None
        self._is_offline = False
        self._checked = False

    def get_driver(self) -> Optional[Driver]:
        if self._is_offline:
            return None
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=5.0,
                    max_connection_lifetime=120.0,
                )
                self._driver.verify_connectivity()
            except Exception as exc:
                logger.warning("Neo4j offline or unreachable (%s): %s", self.uri, exc)
                self._driver = None
                self._is_offline = True
        return self._driver

    def is_connected(self) -> bool:
        if self._is_offline and self._checked:
            return False
        self._checked = True
        return self.get_driver() is not None

    def _mark_offline(self, error: Exception) -> None:
        logger.warning("Neo4j unavailable; continuing without graph writes: %s", error)
        self._is_offline = True
        if self._driver:
            self._driver.close()
            self._driver = None

    def _uid(self, value: str) -> str:
        return f"{self.dataset_id}:{value}"

    def init_constraints(self) -> None:
        driver = self.get_driver()
        if not driver:
            return
        queries = [
            "CREATE CONSTRAINT paper_uid_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.uid IS UNIQUE",
            "CREATE CONSTRAINT topic_uid_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.uid IS UNIQUE",
            "CREATE CONSTRAINT chunk_uid_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE",
            "CREATE CONSTRAINT entity_uid_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
            "CREATE CONSTRAINT mention_uid_unique IF NOT EXISTS FOR (m:Mention) REQUIRE m.uid IS UNIQUE",
        ]
        try:
            with driver.session() as session:
                for query in queries:
                    session.run(query).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def add_document_node(
        self,
        document_name: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a Paper and its extensible Topic nodes.

        The method name is kept for compatibility with the existing pipeline.
        """
        driver = self.get_driver()
        if not driver:
            return
        meta = metadata or {}
        assignments = topic_taxonomy.assign(
            str(meta.get("topic") or ""),
            source=str(meta.get("topic_source") or "metadata"),
        )
        paper_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                session.run(
                    """
                    MERGE (p:Paper {uid: $uid})
                    ON CREATE SET p.created_at = datetime()
                    SET p.paper_id = $paper_id,
                        p.title = $title,
                        p.source = $source,
                        p.dataset_id = $dataset_id,
                        p.schema_version = $schema_version
                    """,
                    uid=paper_uid,
                    paper_id=document_name,
                    title=title,
                    source=meta.get("source", "upload"),
                    dataset_id=self.dataset_id,
                    schema_version=settings.GRAPH_SCHEMA_VERSION,
                ).consume()
                for topic in assignments:
                    session.run(
                        """
                        MATCH (p:Paper {uid: $paper_uid})
                        MERGE (t:Topic:Entity {uid: $topic_uid})
                        ON CREATE SET t.created_at = datetime()
                        SET t.key = $key,
                            t.name = $name,
                            t.entity_id = $entity_id,
                            t.category = 'Topic',
                            t.status = $status,
                            t.source = $topic_source,
                            t.confidence = $confidence,
                            t.dataset_id = $dataset_id,
                            t.schema_version = $schema_version
                        MERGE (t)-[r:HAS_PAPER]->(p)
                        SET r.dataset_id = $dataset_id
                        """,
                        paper_uid=paper_uid,
                        topic_uid=self._uid(f"topic:{topic.key}"),
                        key=topic.key,
                        name=topic.name,
                        entity_id=f"topic:{topic.key}",
                        status=topic.status,
                        topic_source=topic.source,
                        confidence=topic.confidence,
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def add_chunks(self, document_name: str, chunks: List[Any]) -> None:
        """Create Chunk nodes and connect them to the owning Paper."""
        driver = self.get_driver()
        if not driver or not chunks:
            return
        paper_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                for chunk in chunks:
                    session.run(
                        """
                        MATCH (p:Paper {uid: $paper_uid})
                        MERGE (c:Chunk {uid: $chunk_uid})
                        SET c.chunk_id = $chunk_id,
                            c.chunk_index = $chunk_index,
                            c.page = $page,
                            c.text_preview = $text_preview,
                            c.dataset_id = $dataset_id,
                            c.schema_version = $schema_version
                        MERGE (p)-[r:HAS_CHUNK]->(c)
                        SET r.dataset_id = $dataset_id
                        """,
                        paper_uid=paper_uid,
                        chunk_uid=self._uid(chunk.chunk_id),
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        page=chunk.page_number,
                        text_preview=chunk.text[:500],
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def add_entities_and_relations(
        self,
        document_name: str,
        result: GraphExtractionResult,
    ) -> None:
        """Persist canonical entities, mentions, and evidence-backed relations."""
        driver = self.get_driver()
        if not driver:
            return
        paper_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                for entity in result.entities:
                    if not entity.entity_id:
                        logger.warning("Skipping entity without a resolved ID: %s", entity.name)
                        continue
                    entity_uid = self._uid(entity.entity_id)
                    session.run(
                        """
                        MERGE (e:Entity {uid: $uid})
                        ON CREATE SET e.created_at = datetime()
                        SET e.entity_id = $entity_id,
                            e.name = $name,
                            e.category = $category,
                            e.description = $description,
                            e.aliases = $aliases,
                            e.attributes_json = $attributes_json,
                            e.dataset_id = $dataset_id,
                            e.schema_version = $schema_version
                        WITH e
                        MATCH (p:Paper {uid: $paper_uid})
                        MERGE (p)-[r:MENTIONS]->(e)
                        SET r.dataset_id = $dataset_id
                        """,
                        uid=entity_uid,
                        entity_id=entity.entity_id,
                        name=entity.name,
                        category=entity.category,
                        description=entity.description,
                        aliases=entity.aliases,
                        attributes_json=json.dumps(entity.attributes, ensure_ascii=False),
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                        paper_uid=paper_uid,
                    ).consume()

                for mention in result.mentions:
                    if not mention.entity_id or not mention.chunk_id:
                        logger.info("Leaving unresolved mention out of Neo4j: %s", mention.text)
                        continue
                    mention_id = mention.mention_id or hashlib.sha256(
                        f"{mention.chunk_id}|{mention.text}|{mention.entity_id}|{mention.evidence}".encode("utf-8")
                    ).hexdigest()[:24]
                    session.run(
                        """
                        MATCH (c:Chunk {uid: $chunk_uid})
                        MATCH (e:Entity {uid: $entity_uid})
                        MERGE (m:Mention {uid: $mention_uid})
                        SET m.mention_id = $mention_id,
                            m.text = $text,
                            m.mention_type = $mention_type,
                            m.evidence = $evidence,
                            m.confidence = $confidence,
                            m.dataset_id = $dataset_id,
                            m.extraction_version = $extraction_version
                        MERGE (c)-[:CONTAINS_MENTION]->(m)
                        MERGE (m)-[:REFERS_TO]->(e)
                        """,
                        chunk_uid=self._uid(mention.chunk_id),
                        entity_uid=self._uid(mention.entity_id),
                        mention_uid=self._uid(mention_id),
                        mention_id=mention_id,
                        text=mention.text,
                        mention_type=mention.mention_type,
                        evidence=mention.evidence,
                        confidence=mention.confidence,
                        dataset_id=self.dataset_id,
                        extraction_version=settings.EXTRACTION_VERSION,
                    ).consume()

                for relation in result.relations:
                    if relation.relation_type not in ALLOWED_RELATION_TYPES:
                        logger.warning("Rejected unsafe relation type: %s", relation.relation_type)
                        continue
                    if not relation.source_entity_id or not relation.target_entity_id:
                        logger.info(
                            "Skipping unresolved relation: %s -> %s",
                            relation.source,
                            relation.target,
                        )
                        continue
                    evidence_id = hashlib.sha256(
                        f"{relation.chunk_id}|{relation.source_entity_id}|{relation.relation_type}|{relation.target_entity_id}|{relation.evidence}".encode("utf-8")
                    ).hexdigest()[:24]
                    query = f"""
                    MATCH (s:Entity {{uid: $source_uid}})
                    MATCH (t:Entity {{uid: $target_uid}})
                    MERGE (s)-[r:{relation.relation_type} {{evidence_id: $evidence_id}}]->(t)
                    SET r.description = $description,
                        r.evidence = $evidence,
                        r.confidence = $confidence,
                        r.paper_id = $paper_id,
                        r.chunk_id = $chunk_id,
                        r.dataset_id = $dataset_id,
                        r.extraction_version = $extraction_version
                    """
                    session.run(
                        query,
                        source_uid=self._uid(relation.source_entity_id),
                        target_uid=self._uid(relation.target_entity_id),
                        evidence_id=evidence_id,
                        description=relation.description,
                        evidence=relation.evidence,
                        confidence=relation.confidence,
                        paper_id=document_name,
                        chunk_id=relation.chunk_id,
                        dataset_id=self.dataset_id,
                        extraction_version=settings.EXTRACTION_VERSION,
                    ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def get_graph_stats(self) -> Dict[str, Any]:
        driver = self.get_driver()
        if not driver:
            return {"nodes": 0, "edges": 0, "papers": 0, "topics": 0, "status": "offline"}
        try:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (n {dataset_id: $dataset_id})
                    WITH count(n) AS nodes
                    OPTIONAL MATCH ()-[r {dataset_id: $dataset_id}]->()
                    WITH nodes, count(r) AS edges
                    OPTIONAL MATCH (p:Paper {dataset_id: $dataset_id})
                    WITH nodes, edges, count(p) AS papers
                    OPTIONAL MATCH (t:Topic {dataset_id: $dataset_id})
                    RETURN nodes, edges, papers, count(t) AS topics
                    """,
                    dataset_id=self.dataset_id,
                ).single()
                return {**dict(record), "documents": record["papers"], "status": "online"}
        except Exception as exc:
            logger.warning("Could not read graph stats: %s", exc)
            return {"nodes": 0, "edges": 0, "papers": 0, "topics": 0, "status": "error"}

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
graph_builder = GraphBuilder()
