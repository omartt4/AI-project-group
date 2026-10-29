"""
Neo4j Graph Builder module.

Connects to Neo4j knowledge graph database to construct nodes and relationships
for papers, entities, topics, methods, and datasets.
Includes graceful fallback if Neo4j is offline.
"""

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

from graph_rag.config import settings
from graph_rag.graph.entity_extractor import GraphExtractionResult

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Manages Neo4j driver connection and knowledge graph ingestion."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        self._is_offline: bool = False
        self._checked: bool = False

    def get_driver(self) -> Optional[Driver]:
        """Lazy driver connection with retry. Fast failure if offline."""
        if self._is_offline:
            return None

        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=2.0,
                    max_connection_lifetime=60.0,
                )
                self._driver.verify_connectivity()
                self._is_offline = False
            except Exception as e:
                logger.warning(f"Neo4j offline or unreachable ({self.uri}): {e}")
                self._driver = None
                self._is_offline = True
        return self._driver

    def is_connected(self) -> bool:
        """Check if Neo4j is reachable without hanging."""
        if self._is_offline and self._checked:
            return False
        self._checked = True
        driver = self.get_driver()
        return driver is not None

    def init_constraints(self) -> None:
        """Create uniqueness constraints and indexes in Neo4j."""
        driver = self.get_driver()
        if not driver:
            return

        queries = [
            "CREATE CONSTRAINT doc_name_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name_cat IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.category) IS UNIQUE",
        ]
        with driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Constraint query note: {e}")

    def add_document_node(
        self,
        document_name: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update a Document node in the graph."""
        driver = self.get_driver()
        if not driver:
            return

        meta = metadata or {}
        query = """
        MERGE (d:Document {name: $name})
        ON CREATE SET d.title = $title,
                      d.topic = $topic,
                      d.source = $source,
                      d.created_at = datetime()
        ON MATCH SET d.title = $title,
                     d.topic = $topic
        """
        with driver.session() as session:
            session.run(
                query,
                name=document_name,
                title=title,
                topic=meta.get("topic", "General"),
                source=meta.get("source", "upload"),
            )

    def add_entities_and_relations(
        self,
        document_name: str,
        result: GraphExtractionResult,
    ) -> None:
        """Add extracted entities and edges, linking them to the Document node."""
        driver = self.get_driver()
        if not driver or not result.entities:
            return

        with driver.session() as session:
            # 1. Add entities and link to Document
            for entity in result.entities:
                entity_query = """
                MATCH (d:Document {name: $doc_name})
                MERGE (e:Entity {name: $name, category: $category})
                ON CREATE SET e.description = $description
                MERGE (d)-[:MENTIONS]->(e)
                """
                session.run(
                    entity_query,
                    doc_name=document_name,
                    name=entity.name,
                    category=entity.category,
                    description=entity.description,
                )

            # 2. Add relationships between entities
            for rel in result.relations:
                rel_query = f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel.relation_type}]->(t)
                ON CREATE SET r.description = $desc
                """
                try:
                    session.run(
                        rel_query,
                        source=rel.source,
                        target=rel.target,
                        desc=rel.description,
                    )
                except Exception as e:
                    logger.debug(f"Edge creation error: {e}")

    def get_graph_stats(self) -> Dict[str, int]:
        """Return counts of nodes and relationships in Neo4j."""
        driver = self.get_driver()
        if not driver:
            return {"nodes": 0, "edges": 0, "documents": 0, "status": "offline"}

        try:
            with driver.session() as session:
                doc_count = session.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
                node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
                edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "documents": doc_count,
                    "status": "online",
                }
        except Exception:
            return {"nodes": 0, "edges": 0, "documents": 0, "status": "error"}

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


graph_builder = GraphBuilder()
