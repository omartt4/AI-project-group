"""Neo4j writer for the evidence-first graph schema."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from graph_rag.config import settings
from graph_rag.graph.entity_extractor import ALLOWED_RELATION_TYPES, GraphExtractionResult
from graph_rag.graph.topic_taxonomy import (
    SEED_TOPICS,
    TopicAssignment,
    normalize_topic,
    topic_taxonomy,
)

logger = logging.getLogger(__name__)


def _is_byline_only_entity(entity: Any) -> bool:
    """Reject paper-author metadata that is not a body-level graph fact."""
    description = str(entity.description or "").casefold()
    role = str(entity.attributes.get("role", "") or "").casefold()
    if entity.category == "Person":
        return (
            "author of the paper" in description
            or "paper author" in description
            or role in {"author", "co-author", "coauthor"}
        )
    if entity.category == "Organization":
        return (
            description.startswith("affiliation of ")
            or "author affiliation" in description
        )
    return False


class GraphBuilder:
    """Build the compact Paper, Topic, Method, Dataset, Person and Organization graph."""

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
        paper_uid: Optional[str] = None,
    ) -> None:
        """Apply hierarchy labels and links to one canonical topic entity."""
        if assignment.level == "top_level":
            session.run(
                """
                MERGE (t:Entity {uid: $topic_uid})
                SET t:Topic:TopLevelTopic,
                    t.key = $key,
                    t.name = $name,
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

        if paper_uid:
            session.run(
                """
                MATCH (t:Topic {uid: $topic_uid})
                MATCH (p:Paper {uid: $paper_uid})
                MERGE (t)-[r:HAS_PAPER]->(p)
                SET r.dataset_id = $dataset_id
                """,
                topic_uid=topic_uid,
                paper_uid=paper_uid,
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
                paper_parents = tuple(
                    record["name"]
                    for record in session.run(
                        """
                        MATCH (t:Entity {uid: $topic_uid})
                        OPTIONAL MATCH (p1:Paper)-[:MENTIONS]->(t)
                        WITH t, collect(DISTINCT p1) AS mentioned_by
                        OPTIONAL MATCH (t)-[:HAS_PAPER]->(p2:Paper)
                        WITH t, mentioned_by,
                             collect(DISTINCT p2) AS taxonomy_papers
                        WITH t, [p IN mentioned_by + taxonomy_papers
                                WHERE p IS NOT NULL] AS papers
                        UNWIND papers AS paper
                        MATCH (parent:TopLevelTopic)-[:HAS_PAPER]->(paper)
                        WHERE parent.name IN $seed_names AND parent <> t
                        RETURN DISTINCT parent.name AS name
                        ORDER BY name
                        """,
                        topic_uid=row["uid"],
                        seed_names=list(SEED_TOPICS),
                    )
                )
                if (
                    assignment.level == "top_level"
                    and assignment.name not in SEED_TOPICS
                    and paper_parents
                ):
                    assignment = TopicAssignment(
                        name=row["name"],
                        key=normalize_topic(row["name"]).replace(" ", "-"),
                        status="candidate",
                        source="paper_context",
                        confidence=0.7,
                        level="subtopic",
                        parent_names=paper_parents,
                        parent_keys=tuple(
                            normalize_topic(parent).replace(" ", "-")
                            for parent in paper_parents
                        ),
                    )
                self._apply_topic_assignment(
                    session,
                    assignment,
                    topic_uid=row["uid"],
                )
                session.run(
                    """
                    MATCH (p:Paper)-[:MENTIONS]->(t:Topic {uid: $topic_uid})
                    MERGE (t)-[r:HAS_PAPER]->(p)
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
            "DROP CONSTRAINT chunk_uid_unique IF EXISTS",
            "DROP CONSTRAINT mention_uid_unique IF EXISTS",
            "CREATE CONSTRAINT paper_uid_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.uid IS UNIQUE",
            "CREATE CONSTRAINT topic_uid_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.uid IS UNIQUE",
            "CREATE CONSTRAINT entity_uid_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
        ]
        try:
            with driver.session() as session:
                for query in queries:
                    session.run(query).consume()
                session.run(
                    """
                    MATCH (n {dataset_id: $dataset_id})
                    WHERE n:Chunk OR n:Mention
                    DETACH DELETE n
                    """,
                    dataset_id=self.dataset_id,
                ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def add_document_node(
        self,
        document_name: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a Paper and its top-level Topic/Subtopic hierarchy.

        The method name is kept for compatibility with the existing pipeline.
        """
        driver = self.get_driver()
        if not driver:
            return False
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

                # Rebuilding a paper must replace stale taxonomy links without
                # deleting shared topic nodes or any evidence graph data.
                session.run(
                    """
                    MATCH (t:Topic)-[r:HAS_PAPER]->(p:Paper {uid: $paper_uid})
                    DELETE r
                    """,
                    paper_uid=paper_uid,
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
                        MATCH (p:Paper {uid: $paper_uid})
                        MERGE (t:Entity {uid: $topic_uid})
                        ON CREATE SET t.created_at = datetime()
                        SET t:Topic:TopLevelTopic,
                            t.key = $key,
                            t.name = $name,
                            t.entity_id = $entity_id,
                            t.category = 'Topic',
                            t.level = 'top_level',
                            t.status = $status,
                            t.source = $topic_source,
                            t.confidence = $confidence,
                            t.dataset_id = $dataset_id,
                            t.schema_version = $schema_version
                        REMOVE t:Subtopic
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

                for topic in subtopics:
                    session.run(
                        """
                        MATCH (p:Paper {uid: $paper_uid})
                        MERGE (s:Entity {uid: $topic_uid})
                        ON CREATE SET s.created_at = datetime()
                        SET s:Topic:Subtopic,
                            s.key = $key,
                            s.name = $name,
                            s.entity_id = $entity_id,
                            s.category = 'Topic',
                            s.level = 'subtopic',
                            s.status = $status,
                            s.source = $topic_source,
                            s.confidence = $confidence,
                            s.dataset_id = $dataset_id,
                            s.schema_version = $schema_version
                        REMOVE s:TopLevelTopic
                        MERGE (s)-[paper_rel:HAS_PAPER]->(p)
                        SET paper_rel.dataset_id = $dataset_id
                        WITH s
                        UNWIND $parent_uids AS parent_uid
                        MATCH (parent:TopLevelTopic {uid: parent_uid})
                        MERGE (parent)-[taxonomy_rel:HAS_SUBTOPIC]->(s)
                        SET taxonomy_rel.dataset_id = $dataset_id
                        """,
                        paper_uid=paper_uid,
                        topic_uid=self._uid(f"topic:{topic.key}"),
                        key=topic.key,
                        name=topic.name,
                        entity_id=f"topic:{topic.key}",
                        status=topic.status,
                        topic_source=topic.source,
                        confidence=topic.confidence,
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

    def clear_paper_evidence(self, document_name: str) -> None:
        """Remove stale graph evidence for one paper before a graph-only rebuild.

        Shared entities and controlled top-level topics survive when another paper
        still references them. Vector embeddings and local metadata are untouched.
        """
        driver = self.get_driver()
        if not driver:
            return
        paper_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                session.run(
                    """
                    MATCH ()-[r {paper_id: $paper_id}]->()
                    DELETE r
                    """,
                    paper_id=document_name,
                ).consume()
                session.run(
                    """
                    MATCH (p:Paper {uid: $paper_uid})-[r]->(:Entity)
                    WHERE type(r) IN ['MENTIONS', 'CONTAINS_ENTITY', 'USES_METHOD', 'EVALUATES_ON']
                    DELETE r
                    """,
                    paper_uid=paper_uid,
                ).consume()
                session.run(
                    """
                    MATCH (t:Topic)-[r:HAS_PAPER]->(p:Paper {uid: $paper_uid})
                    DELETE r
                    """,
                    paper_uid=paper_uid,
                ).consume()
                session.run(
                    """
                    MATCH (e:Entity {dataset_id: $dataset_id})
                    WHERE NOT e:Topic
                      AND NOT EXISTS {
                          MATCH (:Paper)-[r]->(e)
                          WHERE type(r) IN ['MENTIONS', 'CONTAINS_ENTITY', 'USES_METHOD', 'EVALUATES_ON']
                      }
                    DETACH DELETE e
                    """,
                    dataset_id=self.dataset_id,
                ).consume()
                session.run(
                    """
                    MATCH (t:Topic {dataset_id: $dataset_id})
                    WHERE NOT (t)-[:HAS_PAPER]->(:Paper)
                      AND NOT t.name IN $seed_topics
                    DETACH DELETE t
                    """,
                    dataset_id=self.dataset_id,
                    seed_topics=list(SEED_TOPICS),
                ).consume()
        except Exception as exc:
            self._mark_offline(exc)

    def add_entities_and_relations(
        self,
        document_name: str,
        result: GraphExtractionResult,
    ) -> bool:
        """Persist canonical entities, mentions, and evidence-backed relations."""
        driver = self.get_driver()
        if not driver:
            return False
        paper_uid = self._uid(document_name)
        try:
            with driver.session() as session:
                persisted_uids: Dict[str, str] = {}
                for entity in result.entities:
                    if not entity.entity_id:
                        logger.warning("Skipping entity without a resolved ID: %s", entity.name)
                        continue
                    if _is_byline_only_entity(entity):
                        logger.info("Skipping byline-only %s: %s", entity.category, entity.name)
                        continue
                    entity_uid = self._uid(entity.entity_id)
                    matching_topic = None
                    if entity.category == "Method":
                        matching_topic = session.run(
                            """
                            MATCH (t:Topic {dataset_id: $dataset_id})
                            WHERE toLower(t.name) = toLower($name)
                            RETURN t.uid AS uid
                            LIMIT 1
                            """,
                            dataset_id=self.dataset_id,
                            name=entity.name,
                        ).single()
                    if matching_topic:
                        entity_uid = matching_topic["uid"]
                    else:
                        entity_label = {
                            "Method": "Method",
                            "Dataset": "Dataset",
                            "Person": "Person",
                            "Organization": "Organization",
                        }[entity.category]
                        session.run(
                            f"""
                        MERGE (e:Entity {{uid: $uid}})
                        ON CREATE SET e.created_at = datetime()
                        SET e.entity_id = $entity_id,
                            e.name = $name,
                            e.category = $category,
                            e.description = $description,
                            e.aliases = $aliases,
                            e.attributes_json = $attributes_json,
                            e.role = $role,
                            e.organization = $organization,
                            e.dataset_id = $dataset_id,
                            e.schema_version = $schema_version
                        REMOVE e:Method:Dataset:Person:Organization
                        SET e:Entity:{entity_label}
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
                            dataset_id=self.dataset_id,
                            schema_version=settings.GRAPH_SCHEMA_VERSION,
                        ).consume()
                    persisted_uids[entity.entity_id] = entity_uid
                    if entity.category in {"Person", "Organization"}:
                        session.run(
                            """
                            MATCH (p:Paper {uid: $paper_uid})
                            MATCH (e:Entity {uid: $entity_uid})
                            MERGE (p)-[r:CONTAINS_ENTITY]->(e)
                            SET r.dataset_id = $dataset_id,
                                r.source = 'extraction_contract'
                            """,
                            paper_uid=paper_uid,
                            entity_uid=entity_uid,
                            dataset_id=self.dataset_id,
                        ).consume()
                    paper_relation = {
                        "Method": "USES_METHOD",
                        "Dataset": "EVALUATES_ON",
                    }.get(entity.category)
                    if paper_relation:
                        session.run(
                            f"""
                            MATCH (p:Paper {{uid: $paper_uid}})
                            MATCH (e:Entity {{uid: $entity_uid}})
                            MERGE (p)-[r:{paper_relation}]->(e)
                            SET r.dataset_id = $dataset_id,
                                r.source = 'extraction_contract'
                            """,
                            paper_uid=paper_uid,
                            entity_uid=entity_uid,
                            dataset_id=self.dataset_id,
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
                    source_uid = persisted_uids.get(relation.source_entity_id)
                    target_uid = persisted_uids.get(relation.target_entity_id)
                    if not source_uid or not target_uid:
                        continue
                    query = f"""
                    MATCH (s:Entity {{uid: $source_uid}})
                    MATCH (t:Entity {{uid: $target_uid}})
                    MERGE (s)-[r:{relation.relation_type} {{paper_id: $paper_id}}]->(t)
                    SET r.description = $description,
                        r.evidence = $evidence,
                        r.confidence = CASE
                            WHEN r.confidence IS NULL OR $confidence > r.confidence
                            THEN $confidence ELSE r.confidence END,
                        r.chunk_id = $chunk_id,
                        r.dataset_id = $dataset_id,
                        r.extraction_version = $extraction_version
                    """
                    session.run(
                        query,
                        source_uid=source_uid,
                        target_uid=target_uid,
                        description=relation.description,
                        evidence=relation.evidence,
                        confidence=relation.confidence,
                        paper_id=document_name,
                        chunk_id=relation.chunk_id,
                        dataset_id=self.dataset_id,
                        extraction_version=settings.EXTRACTION_VERSION,
                    ).consume()
            return True
        except Exception as exc:
            self._mark_offline(exc)
            return False

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
                    WHERE toLower(e.role) = toLower($role)
                      AND toLower(e.organization) = toLower($organization)
                    RETURN e.entity_id AS entity_id, e.name AS name,
                           e.description AS description, e.aliases AS aliases,
                           e.role AS role, e.organization AS organization
                    LIMIT 2
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
                "nodes": 0, "edges": 0, "papers": 0, "entities": 0,
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
                    OPTIONAL MATCH (p:Paper {dataset_id: $dataset_id})
                    WITH nodes, edges, count(p) AS papers
                    OPTIONAL MATCH (e:Entity {dataset_id: $dataset_id})
                    WITH nodes, edges, papers, count(e) AS entities,
                         count(CASE WHEN e.category IN
                           ['Person', 'Organization', 'Method', 'Dataset']
                           THEN e END) AS core_entities,
                         count(CASE WHEN e.category = 'Finding' THEN e END) AS findings,
                         count(CASE WHEN e.category = 'Metric' THEN e END) AS metrics
                    OPTIONAL MATCH ()-[m:MENTIONS {dataset_id: $dataset_id}]->()
                    WITH nodes, edges, papers, entities, core_entities, findings,
                         metrics, count(m) AS mentions
                    OPTIONAL MATCH (c:Chunk {dataset_id: $dataset_id})
                    WITH nodes, edges, papers, entities, core_entities, findings,
                         metrics, mentions, count(c) AS chunks
                    OPTIONAL MATCH (t:TopLevelTopic {dataset_id: $dataset_id})
                    WITH nodes, edges, papers, entities, core_entities, findings,
                         metrics, mentions, chunks, count(t) AS topics
                    OPTIONAL MATCH (s:Subtopic {dataset_id: $dataset_id})
                    RETURN nodes, edges, papers, entities, core_entities, findings,
                           metrics, mentions, chunks, topics, count(s) AS subtopics
                    """,
                    dataset_id=self.dataset_id,
                ).single()
                return {**dict(record), "documents": record["papers"], "status": "online"}
        except Exception as exc:
            logger.warning("Could not read graph stats: %s", exc)
            return {
                "nodes": 0, "edges": 0, "papers": 0, "entities": 0,
                "core_entities": 0, "findings": 0, "metrics": 0,
                "mentions": 0, "chunks": 0,
                "topics": 0, "subtopics": 0, "status": "error",
            }

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
graph_builder = GraphBuilder()
