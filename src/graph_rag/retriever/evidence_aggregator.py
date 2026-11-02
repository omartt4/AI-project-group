"""
Evidence Aggregator module for Adaptive Hybrid RAG.

Fuses:
1. Diversity-Aware Vector Excerpts (MMR)
2. Graph-Discovered Concepts & Structural Relationships (Neo4j)
3. Cross-Document Reference Mapping

Produces structured, attributed context formatted specifically for adaptive LLM synthesis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from graph_rag.graph.concept_discovery import ConceptDiscoveryResult
from graph_rag.retriever.query_classifier import QueryClassification, QueryClass

logger = logging.getLogger(__name__)


@dataclass
class AggregatedEvidence:
    """Consolidated and structured evidence package."""
    assembled_context: str
    sources: List[str] = field(default_factory=list)
    discovered_concepts_count: int = 0
    relationships_count: int = 0
    diverse_chunks_count: int = 0
    unique_papers_count: int = 0
    concept_summary: str = ""
    thematic_facets: Dict[str, List[str]] = field(default_factory=dict)


class EvidenceAggregator:
    """Aggregates and formats vector excerpts and graph discoveries into structured prompt context."""

    def aggregate(
        self,
        diverse_chunks: List[Dict[str, Any]],
        discovery: ConceptDiscoveryResult,
        classification: QueryClassification,
    ) -> AggregatedEvidence:
        """
        Synthesize vector passages and graph knowledge into an attributed, structured evidence payload.
        """
        context_sections: List[str] = []
        sources_set: Set[str] = set()
        unique_papers: Set[str] = set()

        # 1. Section: Discovered Knowledge Graph Concepts & Relationships
        has_graph_facts = bool(discovery.relationships or discovery.concepts)
        if has_graph_facts:
            context_sections.append("### 🧠 Discovered Knowledge Graph Concepts & Structural Relationships:")

            # Highlight bridge concepts if available (crucial for comparative queries)
            if discovery.bridge_concepts:
                bridge_names = [f"**{b.name}** ({b.category})" for b in discovery.bridge_concepts]
                context_sections.append(f"- **Connecting Bridge Concepts**: {', '.join(bridge_names)}")

            # Highlight thematic hub concepts
            if discovery.hub_concepts:
                hub_names = [f"**{h.name}** ({h.category})" for h in discovery.hub_concepts]
                context_sections.append(f"- **Core Domain Hubs**: {', '.join(hub_names)}")

            # Highlight categorical clusters
            if discovery.categories_found:
                cat_summaries = []
                for cat, items in discovery.categories_found.items():
                    if items:
                        sample = ", ".join(items[:4])
                        cat_summaries.append(f"{cat}s: [{sample}]")
                if cat_summaries:
                    context_sections.append(f"- **Identified Entities**: {' | '.join(cat_summaries)}")

            # List key structural relationship paths
            if discovery.relationships:
                context_sections.append("- **Knowledge Graph Pathways**:")
                for r in discovery.relationships[:10]:
                    src = r.get("source", "")
                    rel = r.get("relation", "RELATED_TO")
                    tgt = r.get("target", "")
                    context_sections.append(f"  • ({src}) ──[{rel}]──> ({tgt})")

            context_sections.append("")

        # 2. Section: Diversity-Ranked Scientific Passages (MMR)
        if diverse_chunks:
            context_sections.append("### 📚 Diversity-Aware Research Passages (MMR Selected):")
            for idx, c in enumerate(diverse_chunks, start=1):
                meta = c.get("metadata", {})
                doc_name = meta.get("document_name", "Unknown Document")
                doc_title = meta.get("document_title", doc_name)
                page = meta.get("page", 1)
                text = c.get("text", "").strip()
                sim_score = c.get("similarity_score", 0.0)
                div_rank = c.get("diversity_rank", idx)

                unique_papers.add(doc_title)
                source_label = f"{doc_title} (Page {page})"
                sources_set.add(source_label)

                context_sections.append(
                    f"[{idx}] Source: {doc_title} | File: {doc_name} | Page: {page} (Similarity: {sim_score:.2f}, Diversity Rank: #{div_rank})\n"
                    f"{text}\n"
                )
        else:
            if not has_graph_facts:
                context_sections.append("No direct matching documents or graph concepts were found in the knowledge base.")

        assembled_text = "\n".join(context_sections)

        # Build concept summary text for UI / logs
        concept_names = [c.name for c in discovery.concepts[:6]]
        concept_summary = ", ".join(concept_names) if concept_names else "None discovered"

        return AggregatedEvidence(
            assembled_context=assembled_text,
            sources=sorted(list(sources_set)),
            discovered_concepts_count=len(discovery.concepts),
            relationships_count=len(discovery.relationships),
            diverse_chunks_count=len(diverse_chunks),
            unique_papers_count=len(unique_papers),
            concept_summary=concept_summary,
            thematic_facets=discovery.categories_found,
        )


evidence_aggregator = EvidenceAggregator()

