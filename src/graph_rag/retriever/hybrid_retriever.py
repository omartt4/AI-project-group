"""
Hybrid Retriever module for Adaptive Hybrid RAG with Diversity-Aware Retrieval.

Combines:
1. Query Classification & Intent Routing
2. Dynamic Top-K & Diversity-Aware Vector Retrieval (MMR over ChromaDB)
3. Graph-based Concept Discovery (Neo4j multi-hop, bridge & hub exploration)
4. Structured Evidence Aggregation

Maintains 100% backward compatibility with existing baseline retrieve() calls.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.concept_discovery import (
    ConceptDiscoveryResult,
    DiscoveredConcept,
    concept_discovery_engine,
)
from graph_rag.graph.graph_queries import graph_queries
from graph_rag.retriever.diversity_retriever import diversity_retriever
from graph_rag.retriever.evidence_aggregator import (
    AggregatedEvidence,
    evidence_aggregator,
)
from graph_rag.retriever.query_classifier import (
    QueryClassification,
    QueryClass,
    query_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Consolidated context retrieved from vector store and knowledge graph."""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    graph_facts: List[Dict[str, Any]] = field(default_factory=list)
    assembled_text: str = ""
    sources: List[str] = field(default_factory=list)
    classification: Optional[QueryClassification] = None
    discovered_concepts: List[DiscoveredConcept] = field(default_factory=list)
    unique_papers_count: int = 0
    evidence: Optional[AggregatedEvidence] = None


class HybridRetriever:
    """Adaptive Hybrid Retriever combining MMR Vector Search and Graph Concept Discovery."""

    def __init__(self) -> None:
        self.vector_db = vector_store
        self.graph_q = graph_queries
        self.classifier = query_classifier
        self.diversity_retriever = diversity_retriever
        self.concept_engine = concept_discovery_engine
        self.aggregator = evidence_aggregator

    def adaptive_retrieve(
        self,
        query: str,
        classification: Optional[QueryClassification] = None,
    ) -> RetrievedContext:
        """
        Execute Adaptive Hybrid Retrieval with Diversity-Aware MMR and Graph Concept Discovery.
        """
        # 1. Classify Query Intent if not provided
        qc = classification or self.classifier.classify(query)
        logger.info(
            "Adaptive Retrieval started: intent=%s, rec_k=%d, cand_pool=%d, mmr_lambda=%.2f",
            qc.query_class.value,
            qc.recommended_k,
            qc.candidate_pool_size,
            qc.mmr_lambda,
        )

        # Handle purely conversational queries fast
        if qc.query_class == QueryClass.CONVERSATIONAL:
            return RetrievedContext(
                chunks=[],
                graph_facts=[],
                assembled_text="Conversational greeting/interaction.",
                sources=[],
                classification=qc,
                discovered_concepts=[],
                unique_papers_count=0,
            )

        # 2. Diversity-Aware Vector Retrieval (MMR)
        diverse_chunks: List[Dict[str, Any]] = []
        try:
            diverse_chunks = self.diversity_retriever.retrieve_diverse(
                query=query,
                k=qc.recommended_k,
                candidate_k=qc.candidate_pool_size,
                mmr_lambda=qc.mmr_lambda,
            )
        except Exception as e:
            logger.warning(f"Diversity vector retrieval failed: {e}. Falling back to baseline search.")
            try:
                diverse_chunks = self.vector_db.search(query=query, n_results=qc.recommended_k)
            except Exception as e_base:
                logger.error(f"Fallback vector search failed: {e_base}")

        # 3. Graph-based Concept Discovery
        discovery_res = ConceptDiscoveryResult()
        try:
            # Gather search keywords from extracted entities + query terms
            search_keywords = list(qc.extracted_entities)
            if not search_keywords:
                search_keywords = [
                    w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query)
                    if w.lower() not in {"what", "when", "where", "which", "explain", "describe", "show", "tell", "paper", "research"}
                ]

            discovery_res = self.concept_engine.discover(
                keywords=search_keywords,
                comparison_targets=qc.comparison_targets,
                max_depth=qc.graph_depth,
                limit=qc.graph_limit,
            )
        except Exception as e:
            logger.warning(f"Graph concept discovery failed: {e}")

        # 4. Evidence Aggregation & Thematic Structuring
        evidence = self.aggregator.aggregate(
            diverse_chunks=diverse_chunks,
            discovery=discovery_res,
            classification=qc,
        )

        logger.info(
            "Adaptive Retrieval completed: chunks=%d (unique papers=%d), concepts=%d, relations=%d",
            len(diverse_chunks),
            evidence.unique_papers_count,
            len(discovery_res.concepts),
            len(discovery_res.relationships),
        )

        return RetrievedContext(
            chunks=diverse_chunks,
            graph_facts=discovery_res.relationships,
            assembled_text=evidence.assembled_context,
            sources=evidence.sources,
            classification=qc,
            discovered_concepts=discovery_res.concepts,
            unique_papers_count=evidence.unique_papers_count,
            evidence=evidence,
        )

    def retrieve(self, query: str, top_k: int = 5) -> RetrievedContext:
        """
        Baseline retrieval for backward compatibility.
        Routes to adaptive retrieval with requested top_k as override if desired.
        """
        # Execute adaptive retrieval to maintain diversity and concept discovery benefits
        qc = self.classifier.classify(query)
        # Override top_k if explicitly specified by caller
        if top_k != 5:
            qc.recommended_k = top_k
            qc.candidate_pool_size = max(top_k * 2, 12)

        return self.adaptive_retrieve(query, classification=qc)


hybrid_retriever = HybridRetriever()
