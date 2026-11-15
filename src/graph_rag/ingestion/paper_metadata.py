"""Metadata extraction for uploaded research documents."""

import json
import logging
import re
from typing import Dict, List

from graph_rag.ingestion.metadata_manager import infer_topics
from graph_rag.ingestion.pdf_parser import ParsedDocument
from graph_rag.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)


def extract_paper_metadata(document: ParsedDocument) -> Dict[str, str]:
    """Extract paper metadata from its title and text using the configured LLM."""
    source_text = document.full_text[:8000]
    fallback = {
        "title": document.title.strip(),
        "authors": document.metadata.get("author", "").strip() or _extract_authors(document),
        "abstract": source_text[:1000].strip(),
        "topic": infer_topics(document.title, source_text),
        "categories": infer_topics(document.title, source_text),
    }

    prompt = f"""Extract metadata from this research paper.
Return ONLY valid JSON with exactly these string keys:
title, authors, abstract, topic, categories.
Use a semicolon-separated string for authors, topic, and categories.
topic must contain all major subject areas discussed, including multiple topics when applicable.
categories should contain specific research areas, methods, or domains.
Do not invent values. Use an empty string when a value cannot be determined.

Document title: {document.title}
Document text:
{source_text}"""

    try:
        response = ollama_client.generate(
            prompt,
            system_prompt="You are a precise academic metadata extraction assistant.",
            temperature=0.0,
        )
        extracted = _parse_json_response(response)
        if extracted:
            fallback["title"] = extracted.get("title", "").strip() or fallback["title"]
            fallback["authors"] = extracted.get("authors", "").strip() or fallback["authors"]
            fallback["abstract"] = extracted.get("abstract", "").strip() or fallback["abstract"]
            fallback["topic"] = extracted.get("topic", extracted.get("topics", "")).strip() or fallback["topic"]
            fallback["categories"] = extracted.get("categories", "").strip() or fallback["categories"]
    except Exception as exc:
        logger.warning("Structured paper metadata extraction failed; using parser fallback: %s", exc)

    return fallback


def _extract_authors(document: ParsedDocument) -> str:
    """Extract the author block between the title and affiliation lines."""
    lines = [line.strip() for line in document.full_text.splitlines() if line.strip()]
    try:
        title_index = next(index for index, line in enumerate(lines) if document.title[:40] in line)
    except StopIteration:
        return ""

    author_lines: List[str] = []
    for line in lines[title_index + 1 : title_index + 4]:
        if line.lower().startswith(("abstract", "keywords", "1skl", "emails")):
            break
        if re.search(r"[A-Z][a-z]+", line):
            author_lines.append(re.sub(r"[0-9†*‡?,]+", "", line).strip(" ,"))
    return "; ".join(author_lines)


def _parse_json_response(response: str) -> Dict[str, str]:
    """Parse JSON even when a model wraps it in a markdown code fence."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}
