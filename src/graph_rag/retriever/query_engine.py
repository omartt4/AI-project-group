"""
Query Engine module for Adaptive Hybrid RAG with Diversity-Aware Retrieval.

Coordinates:
1. Query Classification (Factual, Comparative, Relational, Exploratory, Summary, Conversational)
2. Dynamic Top-K & Diversity-Aware MMR Retrieval
3. Knowledge Graph Concept & Bridge Discovery
4. Intent-Specific LLM Synthesis (qwen2.5)
5. Stage Latency Instrumentation
6. Real-Time RAG Evaluation (LLM-as-a-Judge)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.config import settings
from graph_rag.evaluation.eval_service import EvaluationRecord, eval_service
from graph_rag.evaluation.latency_tracker import LatencyMetrics, StageTimer
from graph_rag.llm.ollama_client import ollama_client
from graph_rag.retriever.hybrid_retriever import hybrid_retriever
from graph_rag.retriever.query_classifier import QueryClassification, QueryClass
from graph_rag.retriever.language_detector import language_detector

logger = logging.getLogger(__name__)

# Evidence quality thresholds
MIN_CHUNKS_THRESHOLD = 2
MIN_CONTEXT_LENGTH = 100
MIN_SIMILARITY_SCORE = 0.3


def assess_evidence_quality(
    chunks: List[Dict[str, Any]],
    context_text: str,
    classification: Optional[QueryClassification],
) -> tuple[bool, float]:
    """
    Assess whether retrieved evidence is sufficient for answering the question.

    Returns:
        (is_insufficient, quality_score): quality_score ranges 0.0-1.0
    """
    if not chunks:
        return True, 0.0

    # Check chunk count
    chunk_count = len(chunks)
    chunk_score = min(chunk_count / MIN_CHUNKS_THRESHOLD, 1.0)

    # Check context length
    context_length = len(context_text.strip())
    context_score = min(context_length / MIN_CONTEXT_LENGTH, 1.0)

    # Check average similarity score
    similarity_scores = [c.get("similarity_score", 0.0) for c in chunks]
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    similarity_score = min(avg_similarity / MIN_SIMILARITY_SCORE, 1.0)

    # Adjust threshold based on query class
    # Factual queries need higher quality evidence
    if classification and classification.query_class == QueryClass.FACTUAL_LOOKUP:
        weight_chunk = 0.4
        weight_similarity = 0.4
        weight_context = 0.2
    else:
        weight_chunk = 0.3
        weight_similarity = 0.3
        weight_context = 0.4

    quality_score = (
        chunk_score * weight_chunk +
        similarity_score * weight_similarity +
        context_score * weight_context
    )

    # Consider insufficient if quality score is below threshold
    is_insufficient = quality_score < 0.4

    return is_insufficient, round(quality_score, 2)


# Specialized system prompts per query class
SYSTEM_PROMPTS = {
    QueryClass.FACTUAL_LOOKUP: """You are an expert scientific AI research assistant.
Answer the user's factual question concisely, accurately, and strictly based on the retrieved research excerpts.
Cite the exact paper title and page number for each key fact or formula. Avoid speculation.""",

    QueryClass.COMPARATIVE: """You are an expert scientific AI research assistant specializing in comparative research analysis.
Compare and contrast the requested models, architectures, or methods along:
1. Core Architectural Differences
2. Empirical Performance & Benchmarks (datasets, metrics)
3. Strengths, Weaknesses, and Trade-offs
Explicitly mention connecting bridge concepts and cite the relevant research papers with page numbers.""",

    QueryClass.MULTI_HOP_RELATIONAL: """You are an expert scientific AI research assistant with access to a rich Knowledge Graph.
Trace and explain the relational connections (e.g. who proposed what method, which benchmark datasets were evaluated, what underlying techniques were leveraged).
Clearly highlight the knowledge graph pathways and cite the corresponding papers.""",

    QueryClass.SUMMARIZATION: """You are an expert scientific AI research assistant.
Synthesize a comprehensive, well-organized thematic overview across the retrieved research documents.
Structure your answer using thematic headings, summarizing the main directions, methodologies, and findings across distinct papers.
Attribute key findings to their respective paper titles and pages.""",

    QueryClass.EXPLORATORY_THEMATIC: """You are an expert scientific AI research assistant.
Provide a thorough and scientifically rigorous explanation of the topic.
Synthesize insights from the diverse research excerpts and discovered graph concepts.
Ensure proper attribution to the papers and page numbers cited.""",

    QueryClass.CONVERSATIONAL: """You are Graph RAG Studio, a scientific AI research assistant powered by local Ollama models, ChromaDB, and Neo4j.
Respond helpfully, politely, and concisely. If the user asks what you can do, briefly explain that you can answer questions, compare techniques, trace knowledge graph relationships, and synthesize insights across the research paper library.""",
}

INSUFFICIENT_EVIDENCE_PROMPT = """You are a helpful scientific AI research assistant.
The user asked a question, but the retrieved research context is insufficient or irrelevant to provide a reliable answer.

User Question:
{question}

Available Context (limited/irrelevant):
{context}

Guidelines:
1. Acknowledge that the available research documents do not contain sufficient information to answer this question confidently.
2. If possible, suggest what information would be needed or related topics that might be covered in the literature.
3. Be honest about the limitation rather than speculating or hallucinating.
4. Keep the response concise and helpful.

Response:"""


def build_prompts(
    question: str,
    context_text: str,
    classification: Optional[QueryClassification],
) -> tuple[str, str]:
    """Generate adaptive system prompt and user prompt based on query intent."""
    q_class = classification.query_class if classification else QueryClass.EXPLORATORY_THEMATIC
    sys_prompt = SYSTEM_PROMPTS.get(q_class, SYSTEM_PROMPTS[QueryClass.EXPLORATORY_THEMATIC])

    if q_class == QueryClass.CONVERSATIONAL:
        user_prompt = f"User: {question}\n\nAssistant:"
    else:
        user_prompt = (
            f"Research Context & Discovered Knowledge Graph:\n"
            f"{context_text}\n\n"
            f"User Question:\n"
            f"{question}\n\n"
            f"Scientific Answer:"
        )

    return sys_prompt, user_prompt


@dataclass
class QueryEngineResponse:
    """Consolidated response from the Adaptive Hybrid RAG query engine."""
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    context_text: str = ""
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    evaluation: Optional[EvaluationRecord] = None
    classification: Optional[QueryClassification] = None
    discovered_concepts_count: int = 0
    unique_papers_count: int = 0
    concept_summary: str = ""
    insufficient_evidence: bool = False
    evidence_quality_score: float = 1.0
    detected_language: str = "en"


class QueryEngine:
    """Main interface for querying the Adaptive Hybrid RAG system."""

    def __init__(self) -> None:
        self.retriever = hybrid_retriever
        self.llm = ollama_client
        self.eval = eval_service

    def query(
        self,
        question: str,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        message_id: Optional[int] = None,
        run_evaluation: bool = True,
    ) -> QueryEngineResponse:
        """Process user question through the adaptive hybrid RAG pipeline."""
        latency = LatencyMetrics()
        logger.info("Query started: session_id=%s user_id=%s question=%r", session_id, user_id, question[:2000])

        # Stage 1: Adaptive Hybrid Retrieval (Query Classification + Dynamic K + MMR + Graph Discovery)
        with StageTimer() as ret_timer:
            retrieval_res = self.retriever.adaptive_retrieve(query=question)
        latency.retrieval_ms = ret_timer.elapsed_ms

        context_text = retrieval_res.assembled_text
        sources = retrieval_res.sources
        qc = retrieval_res.classification
        concept_summary = retrieval_res.evidence.concept_summary if retrieval_res.evidence else ""

        # Assess evidence quality
        insufficient_evidence, evidence_quality = assess_evidence_quality(
            chunks=retrieval_res.chunks,
            context_text=context_text,
            classification=qc,
        )

        logger.info(
            "Adaptive Retrieval completed: intent=%s, chunks=%d, unique_papers=%d, concepts=%d, relations=%d, latency_ms=%d, evidence_quality=%.2f, insufficient=%s",
            qc.query_class.value if qc else "UNKNOWN",
            len(retrieval_res.chunks),
            retrieval_res.unique_papers_count,
            len(retrieval_res.discovered_concepts),
            len(retrieval_res.graph_facts),
            latency.retrieval_ms,
            evidence_quality,
            insufficient_evidence,
        )

        if len(retrieval_res.discovered_concepts) == 0:
            logger.warning("Zero graph results returned - Neo4j may be offline or no matching entities found")

        # Detect query language for multilingual support
        lang_detection = language_detector.detect(question)
        lang_instruction = language_detector.get_response_language_instruction(lang_detection, question)
        
        logger.info(
            "Language detection: language=%s, confidence=%.2f, script=%s",
            lang_detection.language,
            lang_detection.confidence,
            lang_detection.detected_script,
        )

        # Stage 2: Intent-Specific LLM Generation
        sys_prompt, user_prompt = build_prompts(
            question=question,
            context_text=context_text,
            classification=qc,
        )

        # Add language instruction to system prompt
        sys_prompt = f"{sys_prompt}\n\n{lang_instruction}"

        # Use insufficient evidence prompt if evidence quality is low
        if insufficient_evidence and qc and qc.query_class != QueryClass.CONVERSATIONAL:
            user_prompt = INSUFFICIENT_EVIDENCE_PROMPT.format(
                question=question,
                context=context_text[:500] if context_text else "No relevant context found."
            )
            sys_prompt = f"You are a helpful and honest scientific AI research assistant.\n\n{lang_instruction}"
            logger.warning(f"Insufficient evidence detected (quality={evidence_quality:.2f}), using fallback prompt")

        with StageTimer() as gen_timer:
            try:
                answer = self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=sys_prompt,
                    temperature=0.2 if (qc and qc.query_class != QueryClass.CONVERSATIONAL) else 0.7,
                )
            except Exception as e:
                logger.error(f"LLM Generation failed: {e}")
                answer = f"I encountered an error generating an answer: {e}. Please ensure Ollama is running."
        latency.generation_ms = gen_timer.elapsed_ms

        logger.info(
            "LLM response completed: latency_ms=%d answer=%r",
            latency.generation_ms,
            answer[:4000],
        )

        # Stage 3: Live Evaluation (if enabled and not purely conversational)
        eval_record = None
        should_evaluate = (
            run_evaluation
            and settings.RAG_EVAL_ENABLED
            and (not qc or qc.query_class != QueryClass.CONVERSATIONAL)
        )

        if should_evaluate:
            try:
                eval_record = self.eval.evaluate_turn(
                    question=question,
                    answer=answer,
                    context=context_text,
                    message_id=message_id,
                    session_id=session_id,
                    user_id=user_id,
                    latency=latency,
                    detected_language=lang_detection.language,
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        latency.total_ms = (
            latency.retrieval_ms
            + latency.generation_ms
            + (eval_record.latency.eval_ms if eval_record else 0)
        )

        if eval_record:
            logger.info(
                "Evaluation completed: faithfulness=%.3f answer_relevance=%.3f "
                "context_relevance=%.3f eval_ms=%d total_ms=%d",
                eval_record.faithfulness,
                eval_record.answer_relevance,
                eval_record.context_relevance,
                eval_record.latency.eval_ms,
                latency.total_ms,
            )
        else:
            logger.info("Evaluation skipped: enabled=%s requested=%s intent=%s",
                        settings.RAG_EVAL_ENABLED, run_evaluation, qc.query_class.value if qc else "N/A")

        return QueryEngineResponse(
            question=question,
            answer=answer,
            sources=sources,
            context_text=context_text,
            latency=latency,
            evaluation=eval_record,
            classification=qc,
            discovered_concepts_count=len(retrieval_res.discovered_concepts),
            unique_papers_count=retrieval_res.unique_papers_count,
            concept_summary=concept_summary,
            insufficient_evidence=insufficient_evidence,
            evidence_quality_score=evidence_quality,
            detected_language=lang_detection.language,
        )


query_engine = QueryEngine()
