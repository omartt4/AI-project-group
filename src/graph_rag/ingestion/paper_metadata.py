"""Metadata extraction for uploaded research documents."""

import re
from typing import Dict, List

from graph_rag.ingestion.metadata_manager import infer_topics
from graph_rag.ingestion.pdf_parser import ParsedDocument
def extract_paper_metadata(document: ParsedDocument) -> Dict[str, str]:
    """Extract deterministic metadata without a paid model call."""
    source_text = document.full_text[:8000]
    topics = infer_topics(document.title, source_text)
    return {
        "title": document.title.strip(),
        "authors": document.metadata.get("author", "").strip() or _extract_authors(document),
        "abstract": source_text[:1000].strip(),
        "topic": topics,
        "categories": topics,
    }


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
