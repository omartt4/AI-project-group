"""
Query Classifier module for Adaptive Hybrid RAG.

Classifies incoming queries into semantic intent categories to dynamically determine:
1. Dynamic Top-K budget and candidate pool size
2. MMR diversity lambda (relevance vs diversity trade-off)
3. Graph concept discovery depth and breadth
4. LLM synthesis strategy
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryClass(str, Enum):
    """Semantic intent classification of user query."""
    FACTUAL_LOOKUP = "FACTUAL_LOOKUP"            # Targeted, narrow definition/fact
    COMPARATIVE = "COMPARATIVE"                  # Contrasting two or more entities/methods
    MULTI_HOP_RELATIONAL = "MULTI_HOP_RELATIONAL"  # Relationships, authors, datasets used by methods
    EXPLORATORY_THEMATIC = "EXPLORATORY_THEMATIC"  # Broad topics, approaches, state-of-the-art
    SUMMARIZATION = "SUMMARIZATION"              # Global overview of files, repository, or multiple papers
    CONVERSATIONAL = "CONVERSATIONAL"            # Greetings, gratitude, system capability checks


@dataclass
class QueryClassification:
    """Full classification decision with adaptive hyperparameter recommendations."""
    query_class: QueryClass
    confidence: float
    extracted_entities: List[str] = field(default_factory=list)
    comparison_targets: List[str] = field(default_factory=list)
    recommended_k: int = 5
    candidate_pool_size: int = 15
    mmr_lambda: float = 0.6
    graph_depth: int = 1
    graph_limit: int = 10
    strategy_summary: str = ""


class QueryClassifier:
    """Classifies user queries and computes adaptive retrieval parameters."""

    # Keywords indicative of different query classes
    COMPARISON_PATTERNS = [
        r"\b(?:compare|comparison|versus|vs\.?|difference\s+between|distinction|pros\s+and\s+cons)\b",
        r"\b(?:better\s+than|worse\s+than|advantages?\s+of\s+.+\s+over)\b",
    ]

    SUMMARIZATION_PATTERNS = [
        r"\b(?:summarize|summary|overview|all\s+papers|all\s+files|main\s+topics|across\s+the\s+knowledge\s+base)\b",
        r"\b(?:what\s+papers|list\s+all|corpus|review\s+of|synthesize|landscape)\b",
    ]

    RELATIONAL_PATTERNS = [
        r"\b(?:who\s+authored|written\s+by|published\s+by|proposed\s+by|evaluated\s+on|tested\s+on)\b",
        r"\b(?:(?:which|what)\s+datasets?|used\s+by|applied\s+to|uses?\s+method|related\s+to|connected\s+with|derived\s+from)\b",
        r"\b(?:relationship\s+between|link\s+between|association\s+between)\b",
    ]

    FACTUAL_PATTERNS = [
        r"\b(?:what\s+is|define|definition\s+of|meaning\s+of|formula\s+for|exact|specific)\b",
        r"\b(?:when\s+was|where\s+is|what\s+does\s+.+\s+stand\s+for)\b",
    ]

    CONVERSATIONAL_PATTERNS = [
        r"^(?:hi|hello|hey|greetings|good\s+morning|good\s+evening|good\s+afternoon)(?:\s+(?:there|all|team|assistant|bot|friend))?[\s!.]*$",
        r"^(?:thanks|thank\s+you|thx|cheers|bye|goodbye|see\s+you)[\s!.]*$",
        r"^(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+are\s+you|help|how\s+do\s+you\s+work)[\s!?.]*$",
    ]

    STOP_WORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "the", "and", "or", "for", "with", "about", "into", "from", "that", "this",
        "these", "those", "can", "could", "would", "should", "will", "shall", "may",
        "paper", "papers", "file", "files", "document", "documents", "tell", "show",
        "explain", "give", "please", "using", "used", "does", "have", "been", "were",
        "there", "here", "them", "they", "their", "your", "mine"
    }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify query and determine adaptive hyperparameters.
        Fast pattern and entity matching with fallback logic.
        """
        raw_query = query.strip()
        lower_query = raw_query.lower()

        # 1. Check for Conversational / Greeting
        for pattern in self.CONVERSATIONAL_PATTERNS:
            if re.search(pattern, lower_query):
                return QueryClassification(
                    query_class=QueryClass.CONVERSATIONAL,
                    confidence=0.95,
                    recommended_k=1,
                    candidate_pool_size=3,
                    mmr_lambda=0.9,
                    graph_depth=0,
                    graph_limit=0,
                    strategy_summary="Conversational intent: direct response with minimal retrieval.",
                )

        # Extract potential named entities / key terms
        extracted_entities = self._extract_key_entities(raw_query)

        # 2. Check for Comparative queries
        is_comparative = any(re.search(p, lower_query) for p in self.COMPARISON_PATTERNS)
        comparison_targets: List[str] = []
        if is_comparative or " vs " in lower_query or " versus " in lower_query:
            comparison_targets = self._extract_comparison_targets(raw_query)
            return QueryClassification(
                query_class=QueryClass.COMPARATIVE,
                confidence=0.90 if comparison_targets else 0.80,
                extracted_entities=extracted_entities,
                comparison_targets=comparison_targets,
                recommended_k=7,
                candidate_pool_size=20,
                mmr_lambda=0.50,  # Balanced diversity to cover both comparative targets
                graph_depth=2,    # Look for bridge entities linking both targets
                graph_limit=12,
                strategy_summary="Comparative intent: balanced MMR diversity (λ=0.50) + bridge concept discovery in graph.",
            )

        # 3. Check for Summarization / Broad Landscape queries
        if any(re.search(p, lower_query) for p in self.SUMMARIZATION_PATTERNS):
            return QueryClassification(
                query_class=QueryClass.SUMMARIZATION,
                confidence=0.88,
                extracted_entities=extracted_entities,
                recommended_k=9,
                candidate_pool_size=25,
                mmr_lambda=0.40,  # High diversity across distinct documents
                graph_depth=1,
                graph_limit=18,
                strategy_summary="Summarization intent: high diversity MMR (λ=0.40) across distinct documents + top graph topic hubs.",
            )

        # 4. Check for Multi-Hop / Relational queries
        if any(re.search(p, lower_query) for p in self.RELATIONAL_PATTERNS):
            return QueryClassification(
                query_class=QueryClass.MULTI_HOP_RELATIONAL,
                confidence=0.85,
                extracted_entities=extracted_entities,
                recommended_k=5,
                candidate_pool_size=15,
                mmr_lambda=0.60,
                graph_depth=2,  # Multi-hop graph traversal
                graph_limit=15,
                strategy_summary="Relational intent: 2-hop graph traversal prioritizes connection paths between entities.",
            )

        # 5. Check for Factual / Definition queries
        if any(re.search(p, lower_query) for p in self.FACTUAL_PATTERNS) and len(lower_query.split()) <= 10:
            return QueryClassification(
                query_class=QueryClass.FACTUAL_LOOKUP,
                confidence=0.85,
                extracted_entities=extracted_entities,
                recommended_k=3,
                candidate_pool_size=10,
                mmr_lambda=0.80,  # High precision/relevance focus
                graph_depth=1,
                graph_limit=8,
                strategy_summary="Factual intent: high-precision MMR (λ=0.80, K=3) + 1-hop entity neighborhood.",
            )

        # 6. Default to Exploratory Thematic Query
        # Adjust K slightly based on query length and number of entities
        rec_k = 6 if len(extracted_entities) >= 2 else 5
        cand_size = rec_k * 3
        return QueryClassification(
            query_class=QueryClass.EXPLORATORY_THEMATIC,
            confidence=0.75,
            extracted_entities=extracted_entities,
            recommended_k=rec_k,
            candidate_pool_size=cand_size,
            mmr_lambda=0.55,
            graph_depth=1,
            graph_limit=10,
            strategy_summary=f"Exploratory intent: dynamic K={rec_k} with balanced diversity (λ=0.55) & concept expansion.",
        )

    def _extract_key_entities(self, query: str) -> List[str]:
        """Extract capitalized terms, acronyms, or prominent keywords from query."""
        entities: List[str] = []

        # 1. Find capitalized phrases or acronyms (e.g., "RA-CapNet", "BERT", "Capsule Networks")
        acronyms_and_names = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", query)
        for term in acronyms_and_names:
            if term.lower() not in self.STOP_WORDS and len(term) >= 2:
                entities.append(term)

        # 2. Extract technical terms (hyphenated or alphanumeric)
        hyphenated = re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+\b", query)
        for term in hyphenated:
            if term not in entities and len(term) >= 3:
                entities.append(term)

        # 3. Significant alphanumeric words (> 3 chars not in stop words)
        words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query)
        for w in words:
            w_clean = w.strip()
            if w_clean.lower() not in self.STOP_WORDS and not any(w_clean.lower() in e.lower() for e in entities):
                if len(entities) < 6:
                    entities.append(w_clean)

        return entities[:6]

    def _extract_comparison_targets(self, query: str) -> List[str]:
        """Extract the targets being compared (e.g. 'A vs B' or 'compare A and B')."""
        targets: List[str] = []

        # Pattern 1: A vs B / A versus B
        vs_match = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", query, re.IGNORECASE)
        if vs_match:
            t1 = vs_match.group(1).strip()
            t2 = vs_match.group(2).strip()
            # Clean leading question phrases
            t1 = re.sub(r"^(?:compare|what\s+is\s+the\s+difference\s+between|differences?\s+between)\s+", "", t1, flags=re.IGNORECASE)
            t2 = re.sub(r"[?!.].*$", "", t2).strip()
            if t1:
                targets.append(t1)
            if t2:
                targets.append(t2)
            return targets

        # Pattern 2: compare A and/with B
        comp_match = re.search(r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+)", query, re.IGNORECASE)
        if comp_match:
            t1 = comp_match.group(1).strip()
            t2 = comp_match.group(2).strip()
            t2 = re.sub(r"[?!.].*$", "", t2).strip()
            if t1:
                targets.append(t1)
            if t2:
                targets.append(t2)
            return targets

        return targets


query_classifier = QueryClassifier()
