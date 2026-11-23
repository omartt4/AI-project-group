"""
Graph Queries module.

Provides Cypher query helpers for hybrid graph retrieval.
Searches multi-hop entity neighborhoods and document relationship networks.
"""

from typing import Any, Dict, List, Optional
from graph_rag.graph.graph_builder import graph_builder


class GraphQueries:
    """Cypher query helpers for knowledge graph retrieval."""

    @staticmethod
    def search_entities(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities whose name contains the search keyword."""
        driver = graph_builder.get_driver()
        if not driver:
            return []

        query = """
        MATCH (e:Entity)
        WHERE e.dataset_id = $dataset_id
          AND toLower(e.name) CONTAINS toLower($kw)
        RETURN e.name AS name, e.category AS category, e.description AS description
        LIMIT $limit
        """
        results = []
        try:
            with driver.session() as session:
                for record in session.run(
                    query,
                    kw=keyword,
                    limit=limit,
                    dataset_id=graph_builder.dataset_id,
                ):
                    results.append(
                        {
                            "name": record["name"],
                            "category": record["category"],
                            "description": record["description"] or "",
                        }
                    )
        except Exception:
            pass
        return results

    @staticmethod
    def find_entity_subgraph(entity_name: str, max_depth: int = 1) -> List[Dict[str, Any]]:
        """Retrieve 1-hop connected neighbors and relationships for an entity."""
        driver = graph_builder.get_driver()
        if not driver:
            return []

        query = """
        MATCH (e:Entity {name: $name, dataset_id: $dataset_id})-[r]-(neighbor)
        WHERE neighbor.dataset_id = $dataset_id AND (neighbor:Entity OR neighbor:Paper)
        RETURN e.name AS source, coalesce(r.relation_type, type(r)) AS relation,
               coalesce(neighbor.name, neighbor.title) AS target,
               CASE WHEN neighbor:Paper THEN 'Paper' ELSE neighbor.category END AS target_category
        LIMIT 15
        """
        relationships = []
        try:
            with driver.session() as session:
                for record in session.run(
                    query,
                    name=entity_name,
                    dataset_id=graph_builder.dataset_id,
                ):
                    relationships.append(
                        {
                            "source": record["source"],
                            "relation": record["relation"],
                            "target": record["target"],
                            "target_category": record["target_category"],
                        }
                    )
        except Exception:
            pass
        return relationships

    @staticmethod
    def get_related_documents_for_entities(entity_names: List[str]) -> List[str]:
        """Find papers that mention any of the specified entities."""
        driver = graph_builder.get_driver()
        if not driver or not entity_names:
            return []

        query = """
        MATCH (p:Paper)-[r]->(e:Entity)
        WHERE p.dataset_id = $dataset_id
          AND e.dataset_id = $dataset_id
          AND e.name IN $names
        RETURN DISTINCT p.paper_id AS document_name, p.title AS title
        LIMIT 10
        """
        docs = []
        try:
            with driver.session() as session:
                for record in session.run(
                    query,
                    names=entity_names,
                    dataset_id=graph_builder.dataset_id,
                ):
                    docs.append(record["document_name"])
        except Exception:
            pass
        return docs


graph_queries = GraphQueries()
