"""Structure-aware source preparation for document-level graph extraction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional

from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument


_REFERENCE_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "sources",
    "المراجع",
    "المصادر",
}
_TYPE_HINTS = (
    ("contract", ("agreement", "contract", "lease", "terms and conditions")),
    ("news_article", ("breaking news", "news report", "press release")),
    ("financial_report", ("balance sheet", "income statement", "cash flow statement")),
    ("medical_document", ("patient record", "clinical note", "discharge summary")),
    ("economic_report", ("economic outlook", "economic report", "market outlook")),
    ("research_paper", ("abstract", "methodology", "related work")),
)


@dataclass
class SourceBlock:
    """A compact text region with an exact source-location map."""

    source_ref: str
    block_type: str
    text: str
    page_number: Optional[int] = None
    sheet_name: str = ""
    section: str = ""
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_offsets: List[Dict[str, Any]] = field(default_factory=list)
    character_offsets: List[int] = field(default_factory=list)
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StructuredDocumentInput:
    """Structure-aware complete source for one document-level extraction."""

    document_id: str
    title: str
    document_type: str
    language: str
    blocks: List[SourceBlock]
    spacy_candidates: Dict[str, Any] = field(default_factory=dict)
    known_entity_types: List[str] = field(default_factory=list)
    known_relationship_types: List[str] = field(default_factory=list)
    known_entities: List[Dict[str, Any]] = field(default_factory=list)

    def render(self, blocks: Optional[List[SourceBlock]] = None) -> str:
        selected_blocks = blocks if blocks is not None else self.blocks
        sections = [
            f"Document ID: {self.document_id}",
            f"Title: {self.title}",
            f"Document type: {self.document_type}",
            f"Language: {self.language}",
            "Existing entity types (normalization hints only): "
            + json.dumps(self.known_entity_types, ensure_ascii=False),
            "Existing relationship types (normalization hints only): "
            + json.dumps(self.known_relationship_types, ensure_ascii=False),
            "Known graph entities and context (reuse IDs only with evidence): "
            + json.dumps(self.known_entities, ensure_ascii=False),
            "Source regions:",
        ]
        for block in selected_blocks:
            location = [
                f"ref={block.source_ref}",
                f"type={block.block_type}",
            ]
            if block.page_number is not None:
                location.append(f"page={block.page_number}")
            if block.sheet_name:
                location.append(f"sheet={block.sheet_name}")
            if block.section:
                location.append(f"section={block.section}")
            sections.append(f"[{' | '.join(location)}]\n{block.text}")
        candidate_hints = {
            block.source_ref: {
                "entities": self.spacy_candidates.get(block.source_ref, {}).get("entities", []),
                "noun_phrases": self.spacy_candidates.get(block.source_ref, {}).get("noun_phrases", []),
            }
            for block in selected_blocks
            if block.source_ref in self.spacy_candidates
        }
        if candidate_hints:
            sections.append(
                "spaCy candidate hints (incomplete; never treat as facts):\n"
                + json.dumps(candidate_hints, ensure_ascii=False)
            )
        return "\n\n".join(sections)

    def block_for_evidence(self, source_ref: str, evidence: str) -> Optional[SourceBlock]:
        candidates = [
            block for block in self.blocks
            if not source_ref or block.source_ref == source_ref
        ]
        normalized = " ".join(evidence.casefold().split())
        matches = [
            block for block in candidates
            if normalized and normalized in " ".join(block.text.casefold().split())
        ]
        return matches[0] if len(matches) == 1 else None


def split_document_input(
    document: StructuredDocumentInput,
    max_input_tokens: int,
    token_count: Any,
) -> List[StructuredDocumentInput]:
    """Split only when the complete structured input exceeds its context budget."""
    if token_count(document.render()) <= max_input_tokens:
        return [document]
    groups: List[List[SourceBlock]] = []
    current: List[SourceBlock] = []
    for block in document.blocks:
        single_document = replace(document, blocks=[block])
        parts = [block]
        if token_count(single_document.render()) > max_input_tokens:
            overhead = max(
                1,
                token_count(single_document.render()) - token_count(block.text),
            )
            parts = _split_oversized_block(
                block,
                max(1, max_input_tokens - overhead),
                token_count,
            )
        for part in parts:
            candidate = current + [part]
            if current and token_count(replace(document, blocks=candidate).render()) > max_input_tokens:
                groups.append(current)
                current = [part]
            else:
                current = candidate
            if token_count(replace(document, blocks=current).render()) > max_input_tokens:
                if len(current) == 1:
                    raise ValueError(
                        f"One source region exceeds the configured context budget: {part.source_ref}"
                    )
                groups.append(current[:-1])
                current = [part]
    if current:
        groups.append(current)
    return [
        replace(document, blocks=group)
        for group in groups
        if group
    ]


def _split_oversized_block(
    block: SourceBlock,
    max_input_tokens: int,
    token_count: Any,
) -> List[SourceBlock]:
    """Split a single oversized region at sentences, then at token boundaries."""
    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?؟。])\s+|\n+", block.text)
        if part.strip()
    ]
    if not sentence_parts:
        sentence_parts = [block.text]
    pieces: List[str] = []
    current = ""
    for sentence in sentence_parts:
        candidate = f"{current} {sentence}".strip()
        if current and token_count(candidate) > max_input_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    bounded: List[str] = []
    for piece in pieces:
        if token_count(piece) <= max_input_tokens:
            bounded.append(piece)
            continue
        words = piece.split()
        current_words: List[str] = []
        for word in words:
            candidate_words = current_words + [word]
            if current_words and token_count(" ".join(candidate_words)) > max_input_tokens:
                bounded.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words
        if current_words:
            bounded.append(" ".join(current_words))

    output: List[SourceBlock] = []
    search_from = 0
    for index, text in enumerate(bounded):
        relative_start = block.text.find(text, search_from)
        if relative_start < 0:
            relative_start = search_from
        relative_end = relative_start + len(text)
        char_map = block.character_offsets[relative_start:relative_end]
        output.append(
            replace(
                block,
                source_ref=f"{block.source_ref}:{index + 1}",
                text=text,
                start_offset=(char_map[0] if char_map else block.start_offset),
                end_offset=(char_map[-1] + 1 if char_map else block.end_offset),
                source_offsets=(
                    [{"start": char_map[0], "end": char_map[-1] + 1}]
                    if char_map
                    else block.source_offsets
                ),
                character_offsets=char_map,
            )
        )
        search_from = relative_end
    return output


def _normalize_line(line: str) -> str:
    return re.sub(r"\W+", " ", line.casefold(), flags=re.UNICODE).strip()


def _looks_like_heading(line: str, previous_line: str = "") -> bool:
    value = line.strip()
    if not value or len(value) > 120 or len(value.split()) > 14:
        return False
    if value.startswith("#") or re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", value):
        return True
    if value.endswith(":") and len(value.split()) <= 10:
        return True
    letters = [char for char in value if char.isalpha()]
    if letters and sum(char.isupper() for char in letters) / len(letters) > 0.8:
        return True
    return bool(previous_line.endswith((".", "!", "?", "。")) and len(value.split()) <= 8)


def _detect_document_type(document: ParsedDocument, text: str) -> str:
    explicit = str(document.metadata.get("document_type") or "").strip()
    if explicit:
        return explicit
    sample = f"{document.title}\n{text[:12000]}".casefold()
    scores = [
        (sum(sample.count(hint) for hint in hints), name)
        for name, hints in _TYPE_HINTS
    ]
    score, detected = max(scores, default=(0, "unknown"))
    return detected if score >= 2 else "unknown"


def _source_blocks(
    page: PageContent,
    repeated_lines: set[str],
) -> List[SourceBlock]:
    lines = page.text.splitlines()
    blocks: List[SourceBlock] = []
    section = page.section.strip()
    current: List[str] = []
    current_character_offsets: List[int] = []
    current_start: Optional[int] = None
    offset = 0
    stopped_at_references = False

    def flush(end: int) -> None:
        nonlocal current, current_character_offsets, current_start
        text = "\n".join(current).strip()
        if text:
            start = current_character_offsets[0] if current_character_offsets else current_start
            actual_end = (
                current_character_offsets[-1] + 1
                if current_character_offsets
                else end
            )
            seed = f"{page.page_number}:{section}:{start}:{text}"
            blocks.append(
                SourceBlock(
                    source_ref=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
                    block_type="text",
                    text=text,
                    page_number=page.page_number,
                    sheet_name=page.sheet_name or "",
                    section=section,
                    start_offset=start,
                    end_offset=actual_end,
                    source_offsets=(
                        [{"start": start, "end": actual_end}]
                        if start is not None
                        else page.source_offsets
                    ),
                    character_offsets=list(current_character_offsets),
                    bounding_boxes=list(page.bounding_boxes or []),
                )
            )
        current = []
        current_character_offsets = []
        current_start = None

    previous_line = ""
    for line in lines:
        stripped = line.strip()
        normalized = _normalize_line(stripped)
        if normalized in repeated_lines:
            offset += len(line) + 1
            continue
        if normalized in _REFERENCE_HEADINGS:
            flush(offset)
            stopped_at_references = True
            break
        if _looks_like_heading(stripped, previous_line):
            flush(offset)
            section = stripped.lstrip("#").strip().rstrip(":")
        elif stripped:
            if current_start is None:
                current_start = offset + len(line) - len(line.lstrip())
            if current:
                current_character_offsets.append(offset - 1)
            line_start = offset + len(line) - len(line.lstrip())
            current.append(stripped)
            current_character_offsets.extend(
                range(line_start, line_start + len(stripped))
            )
        previous_line = stripped
        offset += len(line) + 1
    if not stopped_at_references:
        flush(offset)
    return blocks


def prepare_document_input(document: ParsedDocument) -> StructuredDocumentInput:
    """Detect reusable structure and remove repeated boilerplate without sampling chunks."""
    pages = document.pages or [
        PageContent(page_number=1, text=document.full_text, language=document.language)
    ]
    page_lines = [
        [_normalize_line(line) for line in page.text.splitlines() if line.strip()]
        for page in pages
    ]
    repeated = {
        line
        for line in {line for lines in page_lines for line in lines}
        if line and sum(line in lines for lines in page_lines) >= max(2, len(pages) * 0.6)
    }

    blocks: List[SourceBlock] = []
    seen_text: set[str] = set()
    for page in pages:
        for block in _source_blocks(page, repeated):
            key = _normalize_line(block.text)
            if key and key not in seen_text:
                seen_text.add(key)
                blocks.append(block)

    for index, table in enumerate(document.table_data or []):
        rows = table.get("rows") or []
        columns = table.get("columns") or []
        if not rows and not columns:
            continue
        page_number = table.get("page_number")
        sheet_name = str(table.get("sheet_name") or "")
        table_name = str(table.get("table_name") or f"table_{index + 1}")
        text = (
            f"Table: {table_name}\n"
            f"Columns: {json.dumps(columns, ensure_ascii=False, default=str)}\n"
            f"Rows: {json.dumps(rows, ensure_ascii=False, default=str)}"
        )
        seed = f"table:{page_number}:{sheet_name}:{table_name}:{text}"
        blocks.append(
            SourceBlock(
                source_ref=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
                block_type="table",
                text=text,
                page_number=int(page_number) if page_number is not None else None,
                sheet_name=sheet_name,
                section=table_name,
                source_offsets=list(table.get("source_offsets") or []),
                bounding_boxes=list(table.get("bounding_boxes") or []),
            )
        )

    document.document_type = _detect_document_type(
        document,
        "\n".join(block.text for block in blocks),
    )
    document.metadata["document_type"] = document.document_type
    return StructuredDocumentInput(
        document_id=document.document_id or document.content_hash,
        title=document.title,
        document_type=document.document_type,
        language=document.language,
        blocks=blocks,
    )
