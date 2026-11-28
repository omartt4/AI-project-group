"""
LLM-based Query Classifier module.

Uses the local LLM (Ollama) to classify queries into semantic intent categories.
This provides more flexible, multilingual classification compared to regex patterns.
"""

import json
import logging
from typing import Optional

from graph_rag.llm.ollama_client import ollama_client
from graph_rag.retriever.query_classifier import (
    QueryClassification,
    QueryClass,
    query_classifier as regex_classifier,
)

logger = logging.getLogger(__name__)

# LLM prompt for query classification
CLASSIFICATION_PROMPT = """
You are a query classification expert for a scientific research retrieval system.
Classify the user's query into one of these categories:

1. FACTUAL_LOOKUP - Targeted definition/fact questions (e.g., "What is attention mechanism?", "Define cross-entropy")
2. COMPARATIVE - Contrasting two or more entities/methods (e.g., "Compare BERT and GPT", "Difference between CNN and RNN")
3. MULTI_HOP_RELATIONAL - Questions about relationships, authors, datasets (e.g., "Who proposed Transformer?", "What datasets were used to train BERT?")
4. EXPLORATORY_THEMATIC - Broad topics, approaches, state-of-the-art (e.g., "Explain recent advances in computer vision")
5. SUMMARIZATION - Global overview across multiple documents (e.g., "Summarize all papers on attention")
6. CONVERSATIONAL - Greetings, gratitude, system capability checks

Respond in JSON format with this structure:
{
    "query_class": "CATEGORY_NAME",
    "confidence": 0.0-1.0,
    "extracted_entities": ["entity1", "entity2"],
    "comparison_targets": ["target1", "target2"],
    "reasoning": "brief explanation"
}

Query: {query}
"""


class LLMQueryClassifier:
    """LLM-based query classifier with regex fallback."""

    def __init__(self, use_llm: bool = True, fallback_to_regex: bool = True):
        """
        Initialize the LLM classifier.
        
        Args:
            use_llm: Whether to use LLM for classification (default: True)
            fallback_to_regex: Whether to fall back to regex if LLM fails (default: True)
        """
        self.use_llm = use_llm
        self.fallback_to_regex = fallback_to_regex
        self.llm = ollama_client

    def classify(self, query: str) -> QueryClassification:
        """
        Classify query using LLM with regex fallback.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryClassification with adaptive hyperparameters
        """
        # Try LLM classification first
        if self.use_llm:
            try:
                return self._classify_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to regex")
                if not self.fallback_to_regex:
                    # Return default classification if no fallback
                    return self._get_default_classification(query)
        
        # Fallback to regex classifier
        if self.fallback_to_regex:
            logger.debug("Using regex-based classification")
            return regex_classifier.classify(query)
        
        # Default fallback
        return self._get_default_classification(query)

    def _classify_with_llm(self, query: str) -> QueryClassification:
        """Classify query using LLM."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a helpful query classification assistant. Always respond in valid JSON format.",
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle potential markdown code blocks)
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            
            # Map to QueryClass enum
            query_class_str = result.get("query_class", "EXPLORATORY_THEMATIC")
            try:
                query_class = QueryClass(query_class_str)
            except ValueError:
                logger.warning(f"Unknown query class: {query_class_str}, defaulting to EXPLORATORY_THEMATIC")
                query_class = QueryClass.EXPLORATORY_THEMATIC
            
            confidence = result.get("confidence", 0.75)
            extracted_entities = result.get("extracted_entities", [])
            comparison_targets = result.get("comparison_targets", [])
            
            # Compute adaptive hyperparameters based on classification
            return self._compute_hyperparameters(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                comparison_targets=comparison_targets,
                query=query,
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"LLM response: {response}")
            raise

    def _compute_hyperparameters(
        self,
        query_class: QueryClass,
        confidence: float,
        extracted_entities: list,
        comparison_targets: list,
        query: str,
    ) -> QueryClassification:
        """Compute adaptive hyperparameters based on classification."""
        
        # Base hyperparameters per query class
        if query_class == QueryClass.CONVERSATIONAL:
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                recommended_k=1,
                candidate_pool_size=3,
                mmr_lambda=0.9,
                graph_depth=0,
                graph_limit=0,
                strategy_summary="Conversational intent: direct response with minimal retrieval.",
            )
        
        elif query_class == QueryClass.COMPARATIVE:
            # Dynamic K based on comparison targets
            comparison_count = max(2, len(comparison_targets) if comparison_targets else len(extracted_entities))
            comp_k = min(16, comparison_count * 3)
            comp_pool = comp_k * 3
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                comparison_targets=comparison_targets,
                recommended_k=comp_k,
                candidate_pool_size=comp_pool,
                mmr_lambda=0.50,
                graph_depth=2,
                graph_limit=12,
                strategy_summary=f"Comparative intent: dynamic K={comp_k} ({comparison_count} targets × 3) + bridge concept discovery.",
            )
        
        elif query_class == QueryClass.SUMMARIZATION:
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                recommended_k=15,
                candidate_pool_size=45,
                mmr_lambda=0.40,
                graph_depth=2,
                graph_limit=25,
                strategy_summary="Summarization intent: high diversity MMR (λ=0.40, K=15) across distinct documents.",
            )
        
        elif query_class == QueryClass.MULTI_HOP_RELATIONAL:
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                recommended_k=5,
                candidate_pool_size=15,
                mmr_lambda=0.60,
                graph_depth=2,
                graph_limit=15,
                strategy_summary="Relational intent: 2-hop graph traversal prioritizes connection paths.",
            )
        
        elif query_class == QueryClass.FACTUAL_LOOKUP:
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                recommended_k=3,
                candidate_pool_size=10,
                mmr_lambda=0.80,
                graph_depth=1,
                graph_limit=8,
                strategy_summary="Factual intent: high-precision MMR (λ=0.80, K=3) + 1-hop entity neighborhood.",
            )
        
        else:  # EXPLORATORY_THEMATIC (default)
            rec_k = 10 if len(extracted_entities) >= 2 else 8
            cand_size = rec_k * 3
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                extracted_entities=extracted_entities,
                recommended_k=rec_k,
                candidate_pool_size=cand_size,
                mmr_lambda=0.55,
                graph_depth=1,
                graph_limit=10,
                strategy_summary=f"Exploratory intent: dynamic K={rec_k} with balanced diversity (λ=0.55).",
            )

    def _get_default_classification(self, query: str) -> QueryClassification:
        """Return a safe default classification."""
        return QueryClassification(
            query_class=QueryClass.EXPLORATORY_THEMATIC,
            confidence=0.5,
            recommended_k=8,
            candidate_pool_size=24,
            mmr_lambda=0.55,
            graph_depth=1,
            graph_limit=10,
            strategy_summary="Default exploratory classification (LLM unavailable).",
        )


# Global singleton instance
llm_query_classifier = LLMQueryClassifier(
    use_llm=True,
    fallback_to_regex=True,
)
