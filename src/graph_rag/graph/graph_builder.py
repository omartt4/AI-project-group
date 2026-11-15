"""
Neo4j Graph Builder module.

Connects to Neo4j knowledge graph database to construct nodes and relationships
for papers, entities, topics, methods, and datasets.
Includes graceful fallback if Neo4j is offline.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

from graph_rag.config import settings, normalize_neo4j_uri
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
        self._last_connectivity_check = 0.0

    def get_driver(self) -> Optional[Driver]:
        """Return a live driver, periodically retrying after connection failures."""
        if self._driver is not None:
            return self._driver

        now = time.monotonic()
        if self._is_offline and now - self._last_connectivity_check < 5:
            return None

        self._last_connectivity_check = now
        try:
            # Normalize URI for Aura instances (neo4j+s:// -> neo4j+ssc://)
            uri = normalize_neo4j_uri(self.uri)
            
            self._driver = GraphDatabase.driver(
                uri,
                auth=(self.user, self.password),
                connection_timeout=2.0,
                connection_acquisition_timeout=settings.NEO4J_TIMEOUT_SECONDS,
                max_connection_lifetime=60.0,
            )
            self._driver.verify_connectivity()
            self._is_offline = False
        except Exception as e:
            if self._driver:
                self._driver.close()
            logger.warning(f"Neo4j offline or unreachable ({self.uri}): {e}")
            self._driver = None
            self._is_offline = True
        return self._driver

    def is_connected(self) -> bool:
        """Check if Neo4j is reachable without hanging."""
        return self.get_driver() is not None

    def run_query(self, query: str, **parameters: Any) -> List[Dict[str, Any]]:
        """Run a bounded Neo4j transaction and return its records as dictionaries."""
        driver = self.get_driver()
        if not driver:
            return []

        with driver.session() as session:
            with session.begin_transaction(timeout=settings.NEO4J_TIMEOUT_SECONDS) as transaction:
                return [
                    record.data()
                    for record in transaction.run(query, parameters)
                ]

    def init_constraints(self) -> None:
        """Create uniqueness constraints and indexes in Neo4j."""
        queries = [
            "CREATE CONSTRAINT doc_name_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name_cat IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.category) IS UNIQUE",
        ]
        for query in queries:
            try:
                self.run_query(query)
            except Exception as e:
                logger.warning(f"Could not initialize Neo4j constraint: {e}")

    def add_document_node(
        self,
        document_name: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or update a Document node in the graph."""
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
        self.run_query(
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
        if not result.entities:
            return

        for entity in result.entities:
            entity_query = """
            MATCH (d:Document {name: $doc_name})
            MERGE (e:Entity {name: $name, category: $category})
            ON CREATE SET e.description = $description
            MERGE (d)-[:MENTIONS]->(e)
            """
            self.run_query(
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
                self.run_query(
                    rel_query,
                    source=rel.source,
                    target=rel.target,
                    desc=rel.description,
                )
            except Exception as e:
                logger.warning(f"Edge creation error: {e}")

    def get_graph_stats(self) -> Dict[str, int]:
        """Return counts of nodes and relationships in Neo4j."""
        if not self.get_driver():
            return {"nodes": 0, "edges": 0, "documents": 0, "status": "offline"}

        try:
            doc_records = self.run_query("MATCH (d:Document) RETURN count(d) as c")
            node_records = self.run_query("MATCH (n) RETURN count(n) as c")
            edge_records = self.run_query("MATCH ()-[r]->() RETURN count(r) as c")
            return {
                "nodes": node_records[0]["c"],
                "edges": edge_records[0]["c"],
                "documents": doc_records[0]["c"],
                "status": "online",
            }
        except Exception as e:
            logger.error(f"Could not read Neo4j graph stats: {e}")
            return {"nodes": 0, "edges": 0, "documents": 0, "status": "error"}

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


graph_builder = GraphBuilder()
