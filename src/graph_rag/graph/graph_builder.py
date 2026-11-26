"""Neo4j writer for the evidence-first graph schema."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from graph_rag.config import settings
from graph_rag.graph.entity_extractor import GraphExtractionResult
from graph_rag.graph.topic_taxonomy import (
    TopicAssignment,
    normalize_topic,
    topic_taxonomy,
)

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Persist generic documents, canonical entities, topics, and relationships."""

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

    def _apply_topic_assignment(
        self,
        session: Any,
        assignment: TopicAssignment,
        topic_uid: str,
        document_uid: Optional[str] = None,
    ) -> None:
        """Apply hierarchy labels and links to one canonical topic entity."""
        if assignment.level == "top_level":
            session.run(
                """
                MERGE (t:Entity {uid: $topic_uid})
                SET t:Topic:TopLevelTopic,
                    t.key = $key,
                    t.name = $name,
                    t.aliases = reduce(
                        acc = coalesce(t.aliases, []),
                        alias IN $aliases |
                        CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                    ),
                    t.entity_id = $entity_id,
                    t.category = 'Topic',
                    t.level = 'top_level',
                    t.status = $status,
                    t.source = $topic_source,
                    t.confidence = $confidence,
                    t.dataset_id = $dataset_id,
                    t.schema_version = $schema_version
                REMOVE t:Subtopic
                """,
                topic_uid=topic_uid,
                key=assignment.key,
                name=assignment.name,
                aliases=list(assignment.aliases),
                entity_id=f"topic:{assignment.key}",
                status=assignment.status,
                topic_source=assignment.source,
                confidence=assignment.confidence,
                dataset_id=self.dataset_id,
                schema_version=settings.GRAPH_SCHEMA_VERSION,
            ).consume()
        else:
            parent_uids = [
                self._uid(f"topic:{parent_key}")
                for parent_key in assignment.parent_keys
            ]
            session.run(
                """
                MERGE (s:Entity {uid: $topic_uid})
                SET s:Topic:Subtopic,
                    s.key = $key,
                    s.name = $name,
                    s.aliases = reduce(
                        acc = coalesce(s.aliases, []),
                        alias IN $aliases |
                        CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                    ),
                    s.entity_id = $entity_id,
                    s.category = 'Topic',
                    s.level = 'subtopic',
                    s.status = $status,
                    s.source = $topic_source,
                    s.confidence = $confidence,
                    s.dataset_id = $dataset_id,
                    s.schema_version = $schema_version
                REMOVE s:TopLevelTopic
                WITH s
                UNWIND $parents AS parent
                MERGE (t:Entity {uid: parent.uid})
                SET t:Topic:TopLevelTopic,
                    t.key = parent.key,
                    t.name = parent.name,
                    t.entity_id = parent.entity_id,
                    t.category = 'Topic',
                    t.level = 'top_level',
                    t.status = 'active',
                    t.source = 'controlled_parent',
                    t.confidence = 1.0,
                    t.dataset_id = $dataset_id,
                    t.schema_version = $schema_version
                REMOVE t:Subtopic
                MERGE (t)-[r:HAS_SUBTOPIC]->(s)
                SET r.dataset_id = $dataset_id
                """,
                topic_uid=topic_uid,
                key=assignment.key,
                name=assignment.name,
                aliases=list(assignment.aliases),
                entity_id=f"topic:{assignment.key}",
                status=assignment.status,
                topic_source=assignment.source,
                confidence=assignment.confidence,
                parents=[
                    {
                        "uid": parent_uid,
                        "key": parent_key,
                        "name": parent_name,
                        "entity_id": f"topic:{parent_key}",
                    }
                    for parent_uid, parent_key, parent_name in zip(
                        parent_uids,
                        assignment.parent_keys,
                        assignment.parent_names,
                    )
                ],
                dataset_id=self.dataset_id,
                schema_version=settings.GRAPH_SCHEMA_VERSION,
            ).consume()

        if document_uid:
            session.run(
                """
                MATCH (t:Topic {uid: $topic_uid})
                MATCH (d:Document {uid: $document_uid})
                MERGE (t)-[r:HAS_DOCUMENT]->(d)
                SET r.dataset_id = $dataset_id,
                    r.source = $topic_source,
                    r.confidence = $confidence,
                    r.evidence = $evidence
                """,
                topic_uid=topic_uid,
                document_uid=document_uid,
                dataset_id=self.dataset_id,
            ).consume()

    def classify_existing_topics(self) -> int:
        """Classify extracted Topic entities that predate the hierarchy."""
        driver = self.get_driver()
        if not driver:
            return 0
        classified = 0
        with driver.session() as session:
            rows = list(session.run(
                """
                MATCH (t:Entity {dataset_id: $dataset_id})
                WHERE t.category = 'Topic'
                RETURN t.uid AS uid, t.name AS name
                """,
                dataset_id=self.dataset_id,
            ))
            for row in rows:
                assignment = topic_taxonomy.identity_for_entity(row["name"])
                self._apply_topic_assignment(
                    session,
                    assignment,
                    topic_uid=row["uid"],
                )
                session.run(
                    """
                    MATCH (d)-[:MENTIONS]->(t:Topic {uid: $topic_uid})
                    WHERE d:Document OR d:Paper
                    SET d:Document
                    MERGE (t)-[r:HAS_DOCUMENT]->(d)
                    SET r.dataset_id = $dataset_id
                    """,
                    topic_uid=row["uid"],
                    dataset_id=self.dataset_id,
                ).consume()
                classified += 1
        return classified

    def init_constraints(self) -> None:
        driver = self.get_driver()
        if not driver:
            return
        queries = [
            "CREATE CONSTRAINT document_uid_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.uid IS UNIQUE",
            "CREATE CONSTRAINT paper_uid_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.uid IS UNIQUE",
            "CREATE CONSTRAINT topic_uid_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.uid IS UNIQUE",
            "CREATE CONSTRAINT entity_uid_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
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
        document_id: str = "",
        document_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a generic Document node and its extracted topic hierarchy."""
        driver = self.get_driver()
        if not driver:
            return False
        meta = metadata or {}
        document_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                existing_topics = [
                    dict(row)
                    for row in session.run(
                        """
                        MATCH (t:Topic {dataset_id: $dataset_id})
                        OPTIONAL MATCH (parent:TopLevelTopic)-[r:HAS_SUBTOPIC]->(t)
                        WHERE r.document_id IS NULL OR r.document_id <> $document_id
                        RETURN t.name AS name, coalesce(t.aliases, []) AS aliases,
                               t.key AS key,
                               t.level AS level, t.status AS status,
                               collect(DISTINCT parent.name) AS parent_names
                        """,
                        dataset_id=self.dataset_id,
                        document_id=document_id or document_name,
                    )
                ]
                assignments = topic_taxonomy.assign(
                    str(meta.get("topic") or ""),
                    source=str(meta.get("topic_source") or "metadata"),
                    proposals=meta.get("topic_proposals") or [],
                    existing_topics=existing_topics,
                    entity_names=meta.get("entity_names") or [],
                    diagnostics=meta.get("diagnostics"),
                )
                session.run(
                    """
                    MATCH ()-[r:HAS_SUBTOPIC {document_id: $document_id}]->()
                    DELETE r
                    """,
                    document_id=document_id or document_name,
                ).consume()
                legacy = session.run(
                    """
                    MATCH (legacy:Paper {uid: $uid})
                    SET legacy:Document
                    RETURN legacy
                    """,
                    uid=document_uid,
                ).single()
                if legacy:
                    session.run(
                        """
                        MATCH (d:Document {uid: $uid})
                        SET d.document_id = $document_id,
                            d.document_name = $document_name,
                            d.document_type = $document_type,
                            d.title = $title,
                            d.source = $source,
                            d.dataset_id = $dataset_id,
                            d.schema_version = $schema_version
                        """,
                        uid=document_uid,
                        document_id=document_id or document_name,
                        document_name=document_name,
                        document_type=document_type or "unknown",
                        title=title,
                        source=meta.get("source", "upload"),
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()
                else:
                    session.run(
                        """
                        MERGE (d:Document {uid: $uid})
                        ON CREATE SET d.created_at = datetime()
                        SET d.document_id = $document_id,
                            d.document_name = $document_name,
                            d.document_type = $document_type,
                            d.title = $title,
                            d.source = $source,
                            d.dataset_id = $dataset_id,
                            d.schema_version = $schema_version
                        """,
                        uid=document_uid,
                        document_id=document_id or document_name,
                        document_name=document_name,
                        document_type=document_type or "unknown",
                        title=title,
                        source=meta.get("source", "upload"),
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()

                # Rebuilding a document must replace stale taxonomy links without
                # deleting shared topic nodes or any evidence graph data.
                session.run(
                    """
                    MATCH (t:Topic)-[r:HAS_DOCUMENT]->(d:Document {uid: $document_uid})
                    DELETE r
                    """,
                    document_uid=document_uid,
                ).consume()

                top_level_topics = [
                    topic for topic in assignments if topic.level == "top_level"
                ]
                subtopics = [
                    topic for topic in assignments if topic.level == "subtopic"
                ]

                for topic in top_level_topics:
                    session.run(
                        """
                        MATCH (d:Document {uid: $document_uid})
                        MERGE (t:Entity {uid: $topic_uid})
                        ON CREATE SET t.created_at = datetime()
                        SET t:Topic:TopLevelTopic,
                            t.key = $key,
                            t.name = $name,
                            t.aliases = reduce(
                                acc = coalesce(t.aliases, []),
                                alias IN $aliases |
                                CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                            ),
                            t.entity_id = $entity_id,
                            t.category = 'Topic',
                            t.level = 'top_level',
                            t.status = $status,
                            t.source = $topic_source,
                            t.confidence = $confidence,
                            t.dataset_id = $dataset_id,
                            t.schema_version = $schema_version
                        REMOVE t:Subtopic
                        MERGE (t)-[r:HAS_DOCUMENT]->(d)
                        SET r.dataset_id = $dataset_id,
                            r.source = $topic_source,
                            r.confidence = $confidence,
                            r.evidence = $evidence,
                            r.document_id = $document_id
                        """,
                        document_uid=document_uid,
                        document_id=document_id or document_name,
                        topic_uid=self._uid(f"topic:{topic.key}"),
                        key=topic.key,
                        name=topic.name,
                        aliases=list(topic.aliases),
                        entity_id=f"topic:{topic.key}",
                        status=topic.status,
                        topic_source=topic.source,
                        confidence=topic.confidence,
                        evidence=topic.evidence,
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()

                for topic in subtopics:
                    session.run(
                        """
                        MATCH (d:Document {uid: $document_uid})
                        MERGE (s:Entity {uid: $topic_uid})
                        ON CREATE SET s.created_at = datetime()
                        SET s:Topic:Subtopic,
                            s.key = $key,
                            s.name = $name,
                            s.aliases = reduce(
                                acc = coalesce(s.aliases, []),
                                alias IN $aliases |
                                CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                            ),
                            s.entity_id = $entity_id,
                            s.category = 'Topic',
                            s.level = 'subtopic',
                            s.status = $status,
                            s.source = $topic_source,
                            s.confidence = $confidence,
                            s.dataset_id = $dataset_id,
                            s.schema_version = $schema_version
                        REMOVE s:TopLevelTopic
                        MERGE (s)-[paper_rel:HAS_DOCUMENT]->(d)
                        SET paper_rel.dataset_id = $dataset_id,
                            paper_rel.source = $topic_source,
                            paper_rel.confidence = $confidence,
                            paper_rel.evidence = $evidence,
                            paper_rel.document_id = $document_id
                        WITH s
                        UNWIND $parent_uids AS parent_uid
                        MATCH (parent:TopLevelTopic {uid: parent_uid})
                        MERGE (parent)-[taxonomy_rel:HAS_SUBTOPIC
                            {document_id: $document_id}]->(s)
                        SET taxonomy_rel.dataset_id = $dataset_id,
                            taxonomy_rel.source = $topic_source,
                            taxonomy_rel.confidence = $confidence,
                            taxonomy_rel.evidence = $evidence
                        """,
                        document_uid=document_uid,
                        document_id=document_id or document_name,
                        topic_uid=self._uid(f"topic:{topic.key}"),
                        key=topic.key,
                        name=topic.name,
                        aliases=list(topic.aliases),
                        entity_id=f"topic:{topic.key}",
                        status=topic.status,
                        topic_source=topic.source,
                        confidence=topic.confidence,
                        evidence=topic.evidence,
                        parent_uids=[
                            self._uid(f"topic:{parent_key}")
                            for parent_key in topic.parent_keys
                        ],
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()
            return True
        except Exception as exc:
            self._mark_offline(exc)
            return False

    def clear_document_evidence(
        self,
        document_name: str,
        document_id: str = "",
    ) -> None:
        """Remove stale graph evidence while preserving entities shared by other documents."""
        driver = self.get_driver()
        if not driver:
            return
        document_uid = self._uid(document_name)
        stable_document_id = document_id or document_name
        try:
            with driver.session() as session:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.document_id = $document_id
                       OR r.document_name = $document_name
                       OR r.paper_id = $document_name
                    DELETE r
                    """,
                    document_id=stable_document_id,
                    document_name=document_name,
                ).consume()
                session.run(
                    """
                    MATCH (d:Document {uid: $document_uid})-[r]->(:Entity)
                    DELETE r
                    """,
                    document_uid=document_uid,
                ).consume()
                session.run(
                    """
                    MATCH (t:Topic)-[r:HAS_DOCUMENT]->(d:Document {uid: $document_uid})
                    DELETE r
                    """,
                    document_uid=document_uid,
                ).consume()
                session.run(
                    """
                    MATCH (e:Entity {dataset_id: $dataset_id})
                    WHERE NOT e:Topic
                      AND NOT EXISTS { MATCH (:Document)-[]->(e) }
                      AND NOT EXISTS { MATCH (e)-[]->(:Document) }
                      AND NOT EXISTS { MATCH ()-[r]->(e) WHERE r.document_id IS NOT NULL
                                       AND r.document_id <> $document_id }
                      AND NOT EXISTS { MATCH (e)-[r]->() WHERE r.document_id IS NOT NULL
                                       AND r.document_id <> $document_id }
                    DETACH DELETE e
                    """,
                    dataset_id=self.dataset_id,
                    document_id=stable_document_id,
                ).consume()
                session.run(
                    """
                    MATCH (t:Topic {dataset_id: $dataset_id})
                    WHERE NOT (t)-[:HAS_DOCUMENT]->(:Document)
                    DETACH DELETE t
                    """,
                    dataset_id=self.dataset_id,
                ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def clear_paper_evidence(self, document_name: str) -> None:
        """Compatibility wrapper for graph rebuild callers using the former API."""
        self.clear_document_evidence(document_name)

    def delete_document(self, document_name: str, document_id: str = "") -> bool:
        """Delete one generic document and only evidence that depends on it."""
        if not self.get_driver():
            return False
        self.clear_document_evidence(document_name, document_id=document_id)
        driver = self.get_driver()
        if not driver:
            return False
        try:
            with driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {uid: $document_uid, dataset_id: $dataset_id})
                    DETACH DELETE d
                    """,
                    document_uid=self._uid(document_name),
                    dataset_id=self.dataset_id,
                ).consume()
            return True
        except Exception as exc:
            self._mark_offline(exc)
            return False

    def add_entities_and_relations(
        self,
        document_name: str,
        result: GraphExtractionResult,
        document_id: str = "",
    ) -> bool:
        """Persist canonical entities and evidence-backed semantic relationships."""
        driver = self.get_driver()
        if not driver:
            return False
        document_uid = self._uid(document_name)
        stable_document_id = document_id or document_name
        try:
            with driver.session() as session:
                persisted_uids: Dict[str, str] = {}
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
                            e.role = $role,
                            e.organization = $organization,
                            e.evidence = $evidence,
                            e.confidence = $confidence,
                            e.provenance_json = $provenance_json,
                            e.dataset_id = $dataset_id,
                            e.schema_version = $schema_version
                        """,
                        uid=entity_uid,
                        entity_id=entity.entity_id,
                        name=entity.name,
                        category=entity.category,
                        description=entity.description,
                        aliases=entity.aliases,
                        attributes_json=json.dumps(entity.attributes, ensure_ascii=False),
                        role=entity.attributes.get("role", ""),
                        organization=entity.attributes.get("organization", ""),
                        evidence=getattr(entity, "evidence", ""),
                        confidence=getattr(entity, "confidence", 0.0),
                        provenance_json=json.dumps(
                            getattr(entity, "provenance", {}) or {},
                            ensure_ascii=False,
                        ),
                        dataset_id=self.dataset_id,
                        schema_version=settings.GRAPH_SCHEMA_VERSION,
                    ).consume()
                    persisted_uids[entity.entity_id] = entity_uid
                    session.run(
                        """
                        MATCH (d:Document {uid: $document_uid})
                        MATCH (e:Entity {uid: $entity_uid})
                        MERGE (d)-[r:HAS_ENTITY_EVIDENCE {
                            document_id: $document_id,
                            entity_id: $entity_id
                        }]->(e)
                        SET r.evidence = $evidence,
                            r.confidence = $confidence,
                            r.provenance_json = $provenance_json,
                            r.extraction_model = $extraction_model,
                            r.extraction_version = $extraction_version,
                            r.dataset_id = $dataset_id
                        """,
                        document_uid=document_uid,
                        entity_uid=entity_uid,
                        document_id=stable_document_id,
                        entity_id=entity.entity_id,
                        evidence=getattr(entity, "evidence", ""),
                        confidence=getattr(entity, "confidence", 0.0),
                        provenance_json=json.dumps(
                            getattr(entity, "provenance", {}) or {},
                            ensure_ascii=False,
                        ),
                        extraction_model=settings.EXTRACTION_MODEL,
                        extraction_version=settings.EXTRACTION_VERSION,
                        dataset_id=self.dataset_id,
                    ).consume()
                for relation in result.relations:
                    if not relation.source_entity_id or not relation.target_entity_id:
                        logger.info(
                            "Skipping unresolved relation: %s -> %s",
                            relation.source,
                            relation.target,
                        )
                        continue
                    source_uid = persisted_uids.get(relation.source_entity_id)
                    target_uid = persisted_uids.get(relation.target_entity_id)
                    if not source_uid or not target_uid:
                        continue
                    query = """
                    MATCH (s:Entity {uid: $source_uid})
                    MATCH (t:Entity {uid: $target_uid})
                    MERGE (s)-[r:RELATION {document_id: $document_id,
                                           relation_type: $relation_type}]->(t)
                    SET r.description = $description,
                        r.evidence = $evidence,
                        r.confidence = CASE
                            WHEN r.confidence IS NULL OR $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.source_ref = $source_ref,
                        r.provenance_json = $provenance_json,
                        r.document_name = $document_name,
                        r.dataset_id = $dataset_id,
                        r.extraction_version = $extraction_version,
                        r.extraction_model = $extraction_model
                    """
                    session.run(
                        query,
                        source_uid=source_uid,
                        target_uid=target_uid,
                        relation_type=relation.relation_type,
                        description=relation.description,
                        evidence=relation.evidence,
                        confidence=relation.confidence,
                        document_id=stable_document_id,
                        document_name=document_name,
                        source_ref=getattr(relation, "source_ref", ""),
                        provenance_json=json.dumps(
                            getattr(relation, "provenance", {}) or {},
                            ensure_ascii=False,
                        ),
                        dataset_id=self.dataset_id,
                        extraction_version=settings.EXTRACTION_VERSION,
                        extraction_model=settings.EXTRACTION_MODEL,
                    ).consume()
            return True
        except Exception as exc:
            self._mark_offline(exc)
            return False

    @staticmethod
    def _entity_row(row: Any) -> Dict[str, Any]:
        try:
            attributes = json.loads(row.get("attributes_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            attributes = {}
        return {
            "entity_id": row.get("entity_id") or "",
            "name": row.get("name") or "",
            "category": row.get("category") or "",
            "description": row.get("description") or "",
            "aliases": row.get("aliases") or [],
            "attributes": attributes if isinstance(attributes, dict) else {},
        }

    def get_graph_vocabulary(self) -> Dict[str, List[str]]:
        """Return open-schema types already present for retrieval and normalization."""
        driver = self.get_driver()
        if not driver:
            return {"entity_types": [], "relationship_types": []}
        try:
            with driver.session() as session:
                entity_types = [
                    row["category"]
                    for row in session.run(
                        """
                        MATCH (e:Entity {dataset_id: $dataset_id})
                        WHERE e.category IS NOT NULL AND e.category <> 'Topic'
                        RETURN DISTINCT e.category AS category
                        ORDER BY category
                        """,
                        dataset_id=self.dataset_id,
                    )
                ]
                relationship_types = [
                    row["relation_type"]
                    for row in session.run(
                        """
                        MATCH ()-[r:RELATION {dataset_id: $dataset_id}]->()
                        WHERE r.relation_type IS NOT NULL
                        RETURN DISTINCT r.relation_type AS relation_type
                        ORDER BY relation_type
                        """,
                        dataset_id=self.dataset_id,
                    )
                ]
            return {
                "entity_types": entity_types,
                "relationship_types": relationship_types,
            }
        except Exception as exc:
            logger.exception("Could not retrieve graph type vocabulary")
            raise RuntimeError("Graph type vocabulary lookup failed") from exc

    def get_known_entity_context(
        self,
        candidate_names: List[str],
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve exact graph matches and their immediate evidence-backed context."""
        names = list(dict.fromkeys(
            name.strip().casefold()
            for name in candidate_names
            if name and name.strip()
        ))
        if not names:
            return []
        driver = self.get_driver()
        if not driver:
            return []
        try:
            with driver.session() as session:
                rows = list(
                    session.run(
                        """
                        MATCH (e:Entity {dataset_id: $dataset_id})
                        WHERE toLower(e.name) IN $names
                           OR any(alias IN coalesce(e.aliases, [])
                                  WHERE toLower(alias) IN $names)
                        RETURN e.uid AS uid, e.entity_id AS entity_id,
                               e.name AS name, e.category AS category,
                               e.description AS description, e.aliases AS aliases,
                               e.attributes_json AS attributes_json
                        LIMIT $limit
                        """,
                        dataset_id=self.dataset_id,
                        names=names,
                        limit=limit,
                    )
                )
                context: Dict[str, Dict[str, Any]] = {}
                for row in rows:
                    entity = self._entity_row(row)
                    entity["relationships"] = []
                    context[entity["entity_id"]] = entity
                    related_rows = session.run(
                        """
                        MATCH (e:Entity {uid: $uid})-[r:RELATION]-(other:Entity)
                        RETURN other.entity_id AS entity_id, other.name AS name,
                               other.category AS category,
                               other.description AS description,
                               other.aliases AS aliases,
                               other.attributes_json AS attributes_json,
                               r.relation_type AS relation_type,
                               r.evidence AS evidence
                        LIMIT 20
                        """,
                        uid=row["uid"],
                    )
                    entity["relationships"] = [
                        {
                            **self._entity_row(related),
                            "relation_type": related["relation_type"],
                            "evidence": related["evidence"] or "",
                        }
                        for related in related_rows
                    ]
                return list(context.values())
        except Exception as exc:
            logger.warning("Could not retrieve existing entity context: %s", exc)
            return []

    def find_entity_candidates(
        self,
        *,
        name: str,
        category: str,
        aliases: List[str],
        attributes: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Find exact name/alias or role-context matches without guessing."""
        names = list(dict.fromkeys(
            value.strip().casefold()
            for value in [name, *aliases]
            if value and value.strip()
        ))
        role = str(attributes.get("role") or "").strip().casefold()
        organization = str(attributes.get("organization") or "").strip().casefold()
        if not names and not (role and organization):
            return []
        driver = self.get_driver()
        if not driver:
            return []
        try:
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (e:Entity {dataset_id: $dataset_id})
                    WHERE (
                        toLower(e.name) IN $names
                        OR any(alias IN coalesce(e.aliases, [])
                               WHERE toLower(alias) IN $names)
                    )
                    OR (
                        $role <> '' AND $organization <> ''
                        AND toLower(trim(coalesce(e.role, ''))) = $role
                        AND toLower(trim(coalesce(e.organization, ''))) = $organization
                    )
                    RETURN e.entity_id AS entity_id, e.name AS name,
                           e.category AS category, e.description AS description,
                           e.aliases AS aliases, e.attributes_json AS attributes_json
                    LIMIT 50
                    """,
                    dataset_id=self.dataset_id,
                    names=names,
                    role=role,
                    organization=organization,
                )
                return [self._entity_row(row) for row in rows]
        except Exception as exc:
            logger.exception(
                "Could not resolve graph entity candidate %r: %s",
                name,
                exc,
            )
            raise RuntimeError(
                f"Graph entity candidate lookup failed for {name!r}"
            ) from exc

    def find_person_by_role(self, role: str, organization: str) -> Optional[Dict[str, Any]]:
        """Resolve a unique known person from a role and organization."""
        driver = self.get_driver()
        if not driver or not role.strip() or not organization.strip():
            return None
        try:
            with driver.session() as session:
                rows = list(session.run(
                    """
                    MATCH (e:Entity {dataset_id: $dataset_id, category: 'Person'})
                    WHERE toLower(trim(e.role)) = toLower($role)
                      AND toLower(trim(e.organization)) = toLower($organization)
                    RETURN e.entity_id AS entity_id, e.name AS name,
                           e.description AS description, e.aliases AS aliases,
                           e.role AS role, e.organization AS organization
                    """,
                    dataset_id=self.dataset_id,
                    role=role.strip(),
                    organization=organization.strip(),
                ))
            return dict(rows[0]) if len(rows) == 1 else None
        except Exception as exc:
            logger.warning("Role-based person resolution failed: %s", exc)
            return None

    def get_graph_stats(self) -> Dict[str, Any]:
        driver = self.get_driver()
        if not driver:
            return {
                "nodes": 0, "edges": 0, "documents": 0, "papers": 0, "entities": 0,
                "core_entities": 0, "findings": 0, "metrics": 0,
                "mentions": 0, "chunks": 0,
                "topics": 0, "subtopics": 0, "status": "offline",
            }
        try:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (n {dataset_id: $dataset_id})
                    WITH count(n) AS nodes
                    OPTIONAL MATCH ()-[r {dataset_id: $dataset_id}]->()
                    WITH nodes, count(r) AS edges
                    OPTIONAL MATCH (d {dataset_id: $dataset_id})
                    WHERE d:Document OR d:Paper
                    WITH nodes, edges, count(DISTINCT d) AS documents
                    OPTIONAL MATCH (e:Entity {dataset_id: $dataset_id})
                    WITH nodes, edges, documents, count(e) AS entities,
                         count(CASE WHEN e.category <> 'Topic'
                           THEN e END) AS core_entities,
                         count(CASE WHEN e.category = 'Finding' THEN e END) AS findings,
                         count(CASE WHEN e.category = 'Metric' THEN e END) AS metrics
                    OPTIONAL MATCH (t:TopLevelTopic {dataset_id: $dataset_id})
                    WITH nodes, edges, documents, entities, core_entities, findings,
                         metrics, count(t) AS topics
                    OPTIONAL MATCH (s:Subtopic {dataset_id: $dataset_id})
                    RETURN nodes, edges, documents, entities, core_entities, findings,
                           metrics, topics, count(s) AS subtopics
                    """,
                    dataset_id=self.dataset_id,
                ).single()
                return {
                    **dict(record),
                    "papers": record["documents"],
                    "mentions": 0,
                    "chunks": 0,
                    "status": "online",
                }
        except Exception as exc:
            logger.warning("Could not read graph stats: %s", exc)
            return {
                "nodes": 0, "edges": 0, "documents": 0, "papers": 0, "entities": 0,
                "core_entities": 0, "findings": 0, "metrics": 0,
                "mentions": 0, "chunks": 0,
                "topics": 0, "subtopics": 0, "status": "error",
            }

    def get_indexed_documents(self) -> set[str]:
        """Return the set of document IDs or names present in Neo4j."""
        driver = self.get_driver()
        if not driver:
            return set()
        try:
            with driver.session(database=self.database) as session:
                records = session.run(
                    """
                    MATCH (d:Document {dataset_id: $dataset_id})
                    RETURN coalesce(d.document_id, d.name) AS doc_id
                    """,
                    dataset_id=self.dataset_id,
                )
                return {str(record["doc_id"]) for record in records if record["doc_id"]}
        except Exception as exc:
            logger.warning("Could not list Neo4j indexed documents: %s", exc)
            return set()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
graph_builder = GraphBuilder()
