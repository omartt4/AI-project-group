"""
Graph-based Concept Discovery module.

Traverses Neo4j to discover:
1. Concept Neighborhoods: 1-hop and 2-hop related topics, methods, datasets, and authors.
2. Bridge Concepts: Connecting entities that link two comparative targets.
3. Thematic Hubs: Central concepts and clusters relevant to the user query.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_rag.graph.graph_builder import graph_builder

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredConcept:
    """An entity discovered in the knowledge graph with its structural role."""
    name: str
    category: str  # Topic, Method, Dataset, Author, Finding
    description: str = ""
    degree: int = 0
    role: str = "neighbor"  # 'direct', 'neighbor', 'bridge', 'hub'


@dataclass
class ConceptDiscoveryResult:
    """Complete concept discovery outcome for evidence aggregation."""
    concepts: List[DiscoveredConcept] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    bridge_concepts: List[DiscoveredConcept] = field(default_factory=list)
    hub_concepts: List[DiscoveredConcept] = field(default_factory=list)
    categories_found: Dict[str, List[str]] = field(default_factory=dict)


class ConceptDiscoveryEngine:
    """Engine for exploring entity neighborhoods, bridges, and conceptual hubs."""

    def __init__(self, builder=None) -> None:
        self.builder = builder or graph_builder

    def discover(
        self,
        keywords: List[str],
        comparison_targets: Optional[List[str]] = None,
        max_depth: int = 1,
        limit: int = 12,
    ) -> ConceptDiscoveryResult:
        """
        Execute full concept discovery pipeline.
        """
        driver = self.builder.get_driver()
        if not driver:
            return ConceptDiscoveryResult()

        discovered_entities: Dict[str, DiscoveredConcept] = {}
        relationships: List[Dict[str, Any]] = []
        bridge_concepts: List[DiscoveredConcept] = []
        hub_concepts: List[DiscoveredConcept] = []
        seen_edges: Set[str] = set()

        try:
            with driver.session() as session:
                # 1. If comparative query with targets, discover bridge concepts
                if comparison_targets and len(comparison_targets) >= 2:
                    bridges, bridge_rels = self._find_bridge_concepts(
                        session, comparison_targets[0], comparison_targets[1], limit=6
                    )
                    bridge_concepts.extend(bridges)
                    for b in bridges:
                        discovered_entities[b.name] = b
                    for rel in bridge_rels:
                        edge_key = f"{rel['source']}-{rel['relation']}-{rel['target']}"
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            relationships.append(rel)

                # 2. Concept expansion for key terms / entities
                for kw in keywords[:4]:
                    direct_nodes, node_rels = self._expand_entity_neighborhood(
                        session, kw, max_depth=max_depth, limit=limit
                    )
                    for node in direct_nodes:
                        if node.name not in discovered_entities:
                            discovered_entities[node.name] = node

                    for rel in node_rels:
                        edge_key = f"{rel['source']}-{rel['relation']}-{rel['target']}"
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            relationships.append(rel)

                # 3. Discover Thematic Hubs if we have keywords
                if keywords:
                    hubs = self._find_thematic_hubs(session, keywords, limit=5)
                    hub_concepts.extend(hubs)
                    for h in hubs:
                        if h.name not in discovered_entities:
                            discovered_entities[h.name] = h

        except Exception as e:
            logger.warning(f"Concept discovery failed or interrupted: {e}")

        # Group concepts by category
        categories_found: Dict[str, List[str]] = {}
        for c in discovered_entities.values():
            categories_found.setdefault(c.category, []).append(c.name)

        return ConceptDiscoveryResult(
            concepts=list(discovered_entities.values()),
            relationships=relationships[:limit],
            bridge_concepts=bridge_concepts,
            hub_concepts=hub_concepts,
            categories_found=categories_found,
        )

    def _expand_entity_neighborhood(
        self,
        session,
        keyword: str,
        max_depth: int = 1,
        limit: int = 10,
    ) -> Tuple[List[DiscoveredConcept], List[Dict[str, Any]]]:
        """Find matching entities and traverse their connected neighbors."""
        entities: List[DiscoveredConcept] = []
        relationships: List[Dict[str, Any]] = []

        # Find matching entities and their 1-hop / 2-hop neighbors
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($kw)
        OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
        RETURN e.name AS e_name, e.category AS e_cat, e.description AS e_desc,
               type(r) AS rel, neighbor.name AS n_name, neighbor.category AS n_cat,
               neighbor.description AS n_desc
        LIMIT $limit
        """
        records = session.run(query, kw=keyword, limit=limit)
        for row in records:
            e_name = row["e_name"]
            if e_name:
                entities.append(
                    DiscoveredConcept(
                        name=e_name,
                        category=row["e_cat"] or "Topic",
                        description=row["e_desc"] or "",
                        role="direct",
                    )
                )

            rel_type = row["rel"]
            n_name = row["n_name"]
            if rel_type and n_name:
                entities.append(
                    DiscoveredConcept(
                        name=n_name,
                        category=row["n_cat"] or "Concept",
                        description=row["n_desc"] or "",
                        role="neighbor",
                    )
                )
                relationships.append({
                    "source": e_name,
                    "relation": rel_type,
                    "target": n_name,
                    "source_category": row["e_cat"] or "Entity",
                    "target_category": row["n_cat"] or "Entity",
                })

        return entities, relationships

    def _find_bridge_concepts(
        self,
        session,
        target_a: str,
        target_b: str,
        limit: int = 6,
    ) -> Tuple[List[DiscoveredConcept], List[Dict[str, Any]]]:
        """Find bridge entities that connect two distinct comparative targets."""
        bridges: List[DiscoveredConcept] = []
        relationships: List[Dict[str, Any]] = []

        query = """
        MATCH (a:Entity)
        WHERE toLower(a.name) CONTAINS toLower($target_a)
        MATCH (b:Entity)
        WHERE toLower(b.name) CONTAINS toLower($target_b)
        MATCH (a)-[r1]-(bridge:Entity)-[r2]-(b)
        RETURN DISTINCT bridge.name AS bridge_name, bridge.category AS bridge_cat,
                        bridge.description AS bridge_desc,
                        a.name AS a_name, type(r1) AS r1_type,
                        b.name AS b_name, type(r2) AS r2_type
        LIMIT $limit
        """
        records = session.run(query, target_a=target_a, target_b=target_b, limit=limit)
        for row in records:
            b_name = row["bridge_name"]
            if b_name:
                bridges.append(
                    DiscoveredConcept(
                        name=b_name,
                        category=row["bridge_cat"] or "Concept",
                        description=row["bridge_desc"] or "",
                        role="bridge",
                    )
                )
                relationships.append({
                    "source": row["a_name"],
                    "relation": row["r1_type"],
                    "target": b_name,
                })
                relationships.append({
                    "source": b_name,
                    "relation": row["r2_type"],
                    "target": row["b_name"],
                })

        return bridges, relationships

    def _find_thematic_hubs(
        self,
        session,
        keywords: List[str],
        limit: int = 5,
    ) -> List[DiscoveredConcept]:
        """Identify hub nodes (entities with high connectivity) within the topic domain."""
        hubs: List[DiscoveredConcept] = []
        query = """
        MATCH (e:Entity)
        WHERE any(kw in $keywords WHERE toLower(e.name) CONTAINS toLower(kw))
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) AS degree
        ORDER BY degree DESC
        RETURN e.name AS name, e.category AS cat, e.description AS desc, degree
        LIMIT $limit
        """
        try:
            records = session.run(query, keywords=keywords, limit=limit)
            for row in records:
                if row["name"]:
                    hubs.append(
                        DiscoveredConcept(
                            name=row["name"],
                            category=row["cat"] or "Hub",
                            description=row["desc"] or "",
                            degree=row["degree"] or 0,
                            role="hub",
                        )
                    )
        except Exception:
            pass
        return hubs


concept_discovery_engine = ConceptDiscoveryEngine()
