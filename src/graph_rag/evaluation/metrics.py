"""
RAG Evaluation Metrics module.

Implements LLM-as-a-Judge evaluators for:
1. Faithfulness: Is the answer strictly grounded in the retrieved context?
2. Answer Relevance: Does the answer directly address the user's question?
3. Context Relevance: How relevant are the retrieved chunks to the question?
"""

import json
import logging
import math
import re
from typing import List, Optional, Tuple
from graph_rag.llm.model_client import evaluation_model_client

logger = logging.getLogger(__name__)

COMBINED_EVALUATION_PROMPT = """Evaluate this RAG response once across three metrics.
Return JSON only with keys faithfulness, answer_relevance, and context_relevance.
Each key must contain a score from 0.0 to 1.0 and short reasoning.

Question:
{question}

Answer:
{answer}

Retrieved Context:
{context}

Expected shape:
{{"faithfulness": {{"score": 0.9, "reasoning": "..."}},
  "answer_relevance": {{"score": 0.9, "reasoning": "..."}},
  "context_relevance": {{"score": 0.9, "reasoning": "..."}}}}
"""

FAITHFULNESS_PROMPT = """You are an objective AI evaluation judge assessing the faithfulness of an answer.
Task: Determine if every factual statement in the Answer is supported by the provided Context.
Assign a score from 0.0 to 1.0 (1.0 = completely faithful with no hallucinations; 0.0 = completely ungrounded).

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond ONLY with valid JSON:
{{"score": 0.95, "reasoning": "Every claim is supported by excerpt [1] and [2]." }}
"""

ANSWER_RELEVANCE_PROMPT = """You are an objective AI evaluation judge assessing answer relevance.
Task: Determine if the Answer directly and helpfully answers the user's Question.
Assign a score from 0.0 to 1.0 (1.0 = answers the question perfectly and concisely; 0.0 = completely irrelevant).

Question:
{question}

Answer:
{answer}

Respond ONLY with valid JSON:
{{"score": 0.90, "reasoning": "Directly explains the methodology requested in the question." }}
"""

CONTEXT_RELEVANCE_PROMPT = """You are an objective AI evaluation judge assessing context retrieval quality.
Task: Determine how relevant and useful the retrieved Context passages are for answering the Question.
Assign a score from 0.0 to 1.0 (1.0 = context contains all necessary facts; 0.0 = completely irrelevant passages).

Question:
{question}

Retrieved Context:
{context}

Respond ONLY with valid JSON:
{{"score": 0.85, "reasoning": "Top retrieved chunk discusses the exact technique asked about." }}
"""


def _parse_judge_response(
    raw: str,
    default_score: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Extract score and reasoning from LLM judge response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = float(data["score"])
            if not math.isfinite(score):
                raise ValueError("Judge score must be finite")
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", "")).strip()
            return round(score, 2), reasoning
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if default_score is not None:
        num_match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
        score = float(num_match.group(0)) if num_match else default_score
        return round(max(0.0, min(1.0, score)), 2), raw[:200]
    return None, "Evaluation response could not be parsed"


def evaluate_all_metrics(
    question: str,
    answer: str,
    context: str,
) -> dict[str, Tuple[Optional[float], str]]:
    """Evaluate all three metrics in a single paid model request."""
    prompt = COMBINED_EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        context=context[:3500],
    )
    unavailable = {
        "faithfulness": (None, "Evaluation unavailable"),
        "answer_relevance": (None, "Evaluation unavailable"),
        "context_relevance": (None, "Evaluation unavailable"),
    }
    try:
        raw = evaluation_model_client.generate(
            prompt=prompt,
            system_prompt="You are a strict JSON evaluation judge.",
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        if not isinstance(data, dict):
            raise ValueError("Combined evaluation response must be a JSON object")
        result = {}
        for name in unavailable:
            row = data.get(name)
            if not isinstance(row, dict):
                result[name] = unavailable[name]
                continue
            try:
                score = float(row["score"])
                if not math.isfinite(score):
                    raise ValueError("Judge score must be finite")
            except (KeyError, TypeError, ValueError):
                result[name] = unavailable[name]
                continue
            score = max(0.0, min(1.0, score))
            result[name] = (
                round(score, 2),
                str(row.get("reasoning", "Evaluation reasoning unavailable")),
            )
        return result
    except Exception as exc:
        logger.warning("Combined evaluation failed: %s", exc)
        return unavailable


def evaluate_faithfulness(
    question: str,
    answer: str,
    context: str,
) -> Tuple[Optional[float], str]:
    """Score faithfulness (grounding) of the answer against context."""
    prompt = FAITHFULNESS_PROMPT.format(question=question, answer=answer, context=context[:2500])
    try:
        raw = evaluation_model_client.generate(prompt=prompt, system_prompt="You are a strict JSON evaluation judge.", temperature=0.0)
        return _parse_judge_response(raw)
    except Exception as e:
        logger.warning(f"Faithfulness eval failed: {e}")
        return None, "Evaluation unavailable"


def evaluate_answer_relevance(
    question: str,
    answer: str,
) -> Tuple[Optional[float], str]:
    """Score answer relevance to the user question."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    try:
        raw = evaluation_model_client.generate(prompt=prompt, system_prompt="You are a strict JSON evaluation judge.", temperature=0.0)
        return _parse_judge_response(raw)
    except Exception as e:
        logger.warning(f"Answer relevance eval failed: {e}")
        return None, "Evaluation unavailable"


def evaluate_context_relevance(
    question: str,
    context: str,
) -> Tuple[Optional[float], str]:
    """Score relevance of retrieved context to the question."""
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context[:2500])
    try:
        raw = evaluation_model_client.generate(prompt=prompt, system_prompt="You are a strict JSON evaluation judge.", temperature=0.0)
        return _parse_judge_response(raw)
    except Exception as e:
        logger.warning(f"Context relevance eval failed: {e}")
        return None, "Evaluation unavailable"
