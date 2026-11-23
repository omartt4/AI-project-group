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

from langchain_core.runnables import RunnableLambda, RunnableParallel

from graph_rag.config import settings
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

        retrieval_tasks = RunnableParallel(
            vectors=RunnableLambda(
                lambda _: self._retrieve_vectors(query, qc)
            ),
            graph=RunnableLambda(
                lambda _: self._discover_graph(query, qc)
            ),
        )
        task_results = retrieval_tasks.invoke(
            {},
            config={
                "max_concurrency": settings.LANGCHAIN_MAX_CONCURRENCY,
                "run_name": "hybrid-retrieval",
                "metadata": {"intent": qc.query_class.value},
            },
        )
        diverse_chunks = task_results["vectors"]
        discovery_res = task_results["graph"]

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

    def _retrieve_vectors(
        self,
        query: str,
        classification: QueryClassification,
    ) -> List[Dict[str, Any]]:
        try:
            return self.diversity_retriever.retrieve_diverse(
                query=query,
                k=classification.recommended_k,
                candidate_k=classification.candidate_pool_size,
                mmr_lambda=classification.mmr_lambda,
            )
        except Exception as exc:
            logger.warning(
                "Diversity vector retrieval failed: %s. Falling back to baseline search.",
                exc,
            )
            try:
                return self.vector_db.search(
                    query=query,
                    n_results=classification.recommended_k,
                )
            except Exception:
                logger.exception("Fallback vector search failed")
                return []

    def _discover_graph(
        self,
        query: str,
        classification: QueryClassification,
    ) -> ConceptDiscoveryResult:
        search_keywords = list(classification.extracted_entities)
        if not search_keywords:
            search_keywords = [
                word
                for word in re.findall(r"\b[\w-]{3,}\b", query, flags=re.UNICODE)
                if word.casefold() not in self.classifier.STOP_WORDS
            ]
        try:
            return self.concept_engine.discover(
                keywords=search_keywords,
                comparison_targets=classification.comparison_targets,
                max_depth=classification.graph_depth,
                limit=classification.graph_limit,
            )
        except Exception:
            logger.exception("Graph concept discovery failed")
            return ConceptDiscoveryResult()

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
