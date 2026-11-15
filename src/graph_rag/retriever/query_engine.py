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

logger = logging.getLogger(__name__)

# Specialized system prompts per query class (English)
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

# Arabic system prompts per query class
ARABIC_SYSTEM_PROMPTS = {
    QueryClass.FACTUAL_LOOKUP: """أنت مساعد بحث علمي خبير بالذكاء الاصطناعي.
أجب على سؤال المستخدم بدقة واختصار، وبشكل صارم بناءً على مقتطفات البحث المسترجعة.
اذكر عنوان الورقة البحثية ورقم الصفحة لكل حقيقة أو معادلة رئيسية. تجنب التكهنات.""",

    QueryClass.COMPARATIVE: """أنت مساعد بحث علمي خبير متخصص في التحليل المقارن للبحوث.
قارن ووازن بين النماذج أو البنى أو الطرق المطلوبة من حيث:
1. الاختلافات المعمارية الأساسية
2. الأداء التجريبي والمقاييس (مجموعات البيانات، المقاييس)
3. نقاط القوة والضعف والمقايضاته
اذكر بوضوح مفاهيم الجسر الرابطة واذكر الأوراق البحثية ذات الصلة مع أرقام الصفحات.""",

    QueryClass.MULTI_HOP_RELATIONAL: """أنت مساعد بحث علمي خبير لديك إمكانية الوصول إلى رسم بياني معرفي غني.
تتبع واشرح الروابط العلائقية (مثلاً من اقترح أي طريقة، أي مجموعات بيانات معيارية تم تقييمها، ما التقنيات الأساسية تم استخدامها).
أبرز بوضوح مسارات الرسم البياني المعرفي واذكر الأوراق البحثية المقابلة.""",

    QueryClass.SUMMARIZATION: """أنت مساعد بحث علمي خبير.
قم بتوليف نظرة شاملة ومنظمة جيداً على المستويات الموضوعية عبر المستندات البحثية المسترجعة.
هيكل إجابتك باستخدام عناوين موضوعية، ملخصاً الاتجاهات الرئيسية والمنهجيات والنتائج عبر الأوراق المختلفة.
نسب النتائج الرئيسية إلى عناوين الأوراق وصفحاتها المعنية.""",

    QueryClass.EXPLORATORY_THEMATIC: """أنت مساعد بحث علمي خبير.
قدم شرحاً شاملاً وعلمياً دقيقاً للموضوع.
قم بتوليف الرؤى من مقتطفات البحث المتنوعة والمفاهيم المكتشفة في الرسم البياني.
تأكد من الإسناد الصحيح للأوراق البحثية وأرقام الصفحات المذكورة.""",

    QueryClass.CONVERSATIONAL: """أنت Graph RAG Studio، مساعد بحث علمي بالذكاء الاصطناعي مدعوم بنماذج Ollama المحلية وChromaDB وNeo4j.
استجب بشكل مفيد ومهذب ومختصر. إذا سألك المستخدم عما يمكنك فعله، اشرح باختصار أنك يمكنك الإجابة على الأسئلة، ومقارنة التقنيات، وتتبع علاقات الرسم البياني المعرفي، وتوليف الرؤى عبر مكتبة الأوراق البحثية.""",
}


def build_prompts(
    question: str,
    context_text: str,
    classification: Optional[QueryClassification],
) -> tuple[str, str]:
    """Generate adaptive system prompt and user prompt based on query intent and language settings."""
    q_class = classification.query_class if classification else QueryClass.EXPLORATORY_THEMATIC

    # Select language based on Arabic model settings
    use_arabic = settings.ARABIC_MODEL_ENABLED and settings.ARABIC_PROMPT_LANGUAGE == "arabic"
    prompts_dict = ARABIC_SYSTEM_PROMPTS if use_arabic else SYSTEM_PROMPTS
    sys_prompt = prompts_dict.get(q_class, prompts_dict[QueryClass.EXPLORATORY_THEMATIC])

    if q_class == QueryClass.CONVERSATIONAL:
        if use_arabic:
            user_prompt = f"المستخدم: {question}\n\nالمساعد:"
        else:
            user_prompt = f"User: {question}\n\nAssistant:"
    else:
        if use_arabic:
            user_prompt = (
                f"سياق البحث والرسم البياني المعرفي المكتشف:\n"
                f"{context_text}\n\n"
                f"سؤال المستخدم:\n"
                f"{question}\n\n"
                f"الإجابة العلمية:"
            )
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

        logger.info(
            "Adaptive Retrieval completed: intent=%s, chunks=%d, unique_papers=%d, concepts=%d, relations=%d, latency_ms=%d",
            qc.query_class.value if qc else "UNKNOWN",
            len(retrieval_res.chunks),
            retrieval_res.unique_papers_count,
            len(retrieval_res.discovered_concepts),
            len(retrieval_res.graph_facts),
            latency.retrieval_ms,
        )

        # Stage 2: Intent-Specific LLM Generation
        sys_prompt, user_prompt = build_prompts(
            question=question,
            context_text=context_text,
            classification=qc,
        )

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
        )


query_engine = QueryEngine()
