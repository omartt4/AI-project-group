"""Evidence-first entity, mention, and relationship extraction."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangChainDocument
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_neo4j import LLMGraphTransformer
from pydantic import BaseModel, Field, ValidationError

from graph_rag.config import settings
from graph_rag.ingestion.document_structure import StructuredDocumentInput
from graph_rag.llm.json_mode_adapter import JsonModeChatModel
from graph_rag.llm.model_client import (
    ModelUsageCallbackHandler,
    extraction_model_client,
)

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Canonical, evidence-grounded entity extracted from a document."""

    name: str
    category: str
    description: str = ""
    entity_id: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    evidence: str = ""
    confidence: float = 0.0
    source_ref: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedMention:
    """A source phrase that refers to an entity."""

    text: str
    entity_name: str = ""
    entity_category: str = ""
    entity_id: str = ""
    mention_type: str = "name"
    attributes: Dict[str, str] = field(default_factory=dict)
    evidence: str = ""
    confidence: float = 0.0
    chunk_id: str = ""
    mention_id: str = ""
    source_ref: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """A typed relationship with source evidence and resolved endpoints."""

    source: str
    target: str
    relation_type: str
    description: str = ""
    source_category: str = ""
    target_category: str = ""
    source_entity_id: str = ""
    target_entity_id: str = ""
    evidence: str = ""
    confidence: float = 0.0
    chunk_id: str = ""
    source_ref: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphExtractionResult:
    """Document facts, topic proposals, and non-blocking extraction diagnostics."""

    entities: List[ExtractedEntity] = field(default_factory=list)
    mentions: List[ExtractedMention] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    topics: List["ExtractedTopic"] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    model_responses: List[Dict[str, Any]] = field(default_factory=list)
    cache_metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: str = "unknown"
    cache_stage: str = "ready"


class GraphExtractionError(RuntimeError):
    """Extraction failure that retains any provider responses for diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        model_responses: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.model_responses = model_responses or []


@dataclass
class ExtractedTopic:
    """A topic candidate grounded in document evidence."""

    name: str
    parent_names: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.0
    source_ref: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


class EntityOutput(BaseModel):
    entity_id: str = ""
    name: str
    category: str
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    attributes: Dict[str, str] = Field(default_factory=dict)
    evidence: str = ""
    confidence: float = 0.0
    source_ref: str = ""


class MentionOutput(BaseModel):
    text: str
    entity_id: str = ""
    entity_name: str = ""
    entity_category: str = ""
    mention_type: str = "name"
    attributes: Dict[str, str] = Field(default_factory=dict)
    evidence: str = ""
    confidence: float = 0.0
    source_ref: str = ""


class RelationOutput(BaseModel):
    source_entity_id: str = ""
    source: str
    source_category: str = ""
    target_entity_id: str = ""
    target: str
    target_category: str = ""
    relation_type: str
    description: str = ""
    evidence: str = ""
    confidence: float = 0.0
    source_ref: str = ""


class TopicOutput(BaseModel):
    name: str
    parent_names: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    evidence: str
    confidence: float = 0.0
    source_ref: str = ""


class GraphExtractionOutput(BaseModel):
    document_type: str = "unknown"
    entities: List[EntityOutput] = Field(default_factory=list)
    mentions: List[MentionOutput] = Field(default_factory=list)
    relations: List[RelationOutput] = Field(default_factory=list)
    topics: List[TopicOutput] = Field(default_factory=list)


EXTRACTION_PROMPT_TEMPLATE = """Extract an evidence-first, open-schema knowledge graph from this document region.

Document title: {document_title}
Source reference: {chunk_id}

Known entities from this document:
{known_entities}

Rules:
1. Discover entities, their types, relationships, relationship types, aliases,
   properties, topics, subtopics, and document type from any domain.
2. Do not use a fixed type, relationship, topic, or section taxonomy. Existing
   types supplied in context are normalization hints only; keep valid new types.
3. spaCy output is incomplete candidate assistance, never a fact or a reason
   to reject source-supported information.
4. Every entity and relationship must include an exact, short source quote,
   source_ref, and confidence from 0 to 1. Never invent locations or offsets.
5. Do not turn full sentences, unresolved pronouns, repeated boilerplate,
   headers/footers, reference-list fragments, or meaningless generic wording
   into entities or topics.
6. If a reference is ambiguous, do not guess an identity or topic parent.
   Return its evidence with an empty entity_id or omit the uncertain link.
7. Reuse supplied entity IDs only when the evidence supports the identity.
8. Use aliases for abbreviations and alternate names. Keep entity properties
   open-ended and grounded in evidence.
9. Topic and subtopic names are open-schema concepts, not entity-type labels.
   Include a parent only when the source supports that hierarchy.
10. Use concise discovered node types and relationship types. Relationship
    types may be normalized to UPPER_SNAKE_CASE.

Return JSON only:
{{
  "document_type": "discovered type or unknown",
  "entities": [{{"entity_id": "existing ID or empty", "name": "canonical name", "category": "discovered type", "description": "disambiguating description", "aliases": ["alternate name"], "attributes": {{"role": "stated role"}}, "evidence": "exact quote", "source_ref": "region reference", "confidence": 0.0}}],
  "mentions": [{{"text": "exact source phrase", "entity_id": "known ID when resolved", "entity_name": "canonical name", "entity_category": "discovered type", "mention_type": "name|alias|pronoun|role|other", "attributes": {{}}, "evidence": "exact quote", "source_ref": "region reference", "confidence": 0.0}}],
  "relations": [{{"source_entity_id": "known ID when available", "source": "canonical source name", "source_category": "discovered type", "target_entity_id": "known ID when available", "target": "canonical target name", "target_category": "discovered type", "relation_type": "discovered relationship", "description": "short explanation", "evidence": "exact quote", "source_ref": "region reference", "confidence": 0.0}}],
  "topics": [{{"name": "open-schema topic", "parent_names": ["supported broader topic"], "aliases": [], "evidence": "exact quote", "source_ref": "region reference", "confidence": 0.0}}]
}}

Document region:
{text}
"""

GRAPH_TRANSFORMER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Build an evidence-first, open-schema knowledge graph from any document domain.
The source contains document type, structure, exact region references, and optional
spaCy candidate hints. spaCy labels are incomplete suggestions, never a schema.
Discover novel entity types, semantic relationship types, aliases, properties,
topics, subtopics, and document type. Existing type labels are hints only.

Return the source document as one node with id equal to the provided Document ID,
type Document, and property document_type set to your best evidence-based
classification or unknown. Do not treat this base node as an entity or topic.

Every semantic entity node must have properties evidence (an exact short quote
from one source region), source_ref (that region's exact reference), confidence
(0-1), aliases (JSON array text), and attributes (JSON object text). Every semantic
relationship must have exact evidence, source_ref, confidence, and a concise
description. Use properties as key/value strings.

Return each distinct semantic fact once. Keep evidence to the shortest exact
quote that proves the fact, keep descriptions to one short sentence, and omit
decorative or repeated properties. Prefer a smaller complete JSON object over
repeating the same entity, relationship, alias, mention, or evidence.

Every semantic node must also have a name property. Keep name in the dominant
language and script used by the source evidence. Put translated, transliterated,
abbreviated, and alternate names in aliases. The node id is a stable internal
identifier and must not be used as a substitute for the source-language name.

Represent topics as nodes of type Topic with evidence, source_ref, confidence,
and aliases. Represent a supported parent-child topic hierarchy with a
PARENT_OF relationship between Topic nodes. Do not force a hierarchy.

Reject unsupported facts, generic meaningless phrases, full sentences as entity
names, headers, footers, reference-list fragments, unresolved pronouns, and
low-confidence guesses. Never invent locations, names, aliases, or evidence.
If identity or parentage is ambiguous, omit the link and leave diagnostic
evidence in a property when possible. Keep all valid types open-ended.

For relationships, source_node_id and target_node_id must exactly match node
ids. Return a JSON object with "nodes" and "relationships" arrays. Each node
has "id", "type", and "properties"; each relationship has "source_node_id",
"source_node_type", "target_node_id", "target_node_type", "type", and
"properties". Properties are arrays of {{"key": "...", "value": "..."}} strings.
Every property entry must contain both a key and a non-null string value.
Never nest the top-level relationships array inside a node or property. Use
arbitrary property keys only for grounded data.""",
        ),
        ("human", "{input}"),
    ]
)


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _category(value: Any) -> str:
    raw = " ".join(str(value or "").split())
    if len(raw) > 80 or not any(character.isalnum() for character in raw):
        return ""
    return raw.title()


def _relation_type(value: Any) -> str:
    raw = re.sub(
        r"[^\w]+",
        "_",
        str(value or "").strip().upper(),
        flags=re.UNICODE,
    ).strip("_")
    if len(raw) > 80 or not any(character.isalnum() for character in raw):
        return ""
    return raw


def _grounded(evidence: str, source_text: Optional[str]) -> bool:
    if not evidence:
        return False
    if source_text is None:
        return True
    normalized_evidence, _, _ = _normalize_evidence_text(evidence)
    normalized_source, _, _ = _normalize_evidence_text(source_text)
    return normalized_evidence in normalized_source


def _normalize_evidence_text(text: str) -> tuple[str, List[int], List[int]]:
    """Normalize Arabic spelling variants while mapping matches to original offsets."""
    normalized: List[str] = []
    source_starts: List[int] = []
    source_ends: List[int] = []

    for source_index, character in enumerate(text):
        for value in unicodedata.normalize("NFKC", character):
            codepoint = ord(value)
            is_arabic_mark = (
                0x064B <= codepoint <= 0x065F
                or codepoint == 0x0670
                or 0x06D6 <= codepoint <= 0x06ED
                or 0x08D3 <= codepoint <= 0x08FF
            )
            if value == "\u0640" or is_arabic_mark:
                if source_ends:
                    source_ends[-1] = source_index + 1
                continue

            if value in {"أ", "إ", "آ", "ٱ"}:
                value = "ا"
            elif value == "ى":
                value = "ي"

            if value.isspace():
                if not normalized:
                    continue
                if normalized[-1] == " ":
                    source_ends[-1] = source_index + 1
                else:
                    normalized.append(" ")
                    source_starts.append(source_index)
                    source_ends.append(source_index + 1)
                continue

            for folded in value.casefold():
                normalized.append(folded)
                source_starts.append(source_index)
                source_ends.append(source_index + 1)

    while normalized and normalized[-1] == " ":
        normalized.pop()
        source_starts.pop()
        source_ends.pop()
    return "".join(normalized), source_starts, source_ends


_REJECTED_NAME_PATTERNS = (
    re.compile(r"^(?:header|footer|page\s+\d+)\b", re.IGNORECASE),
    re.compile(r"\bet\s+al\.?$", re.IGNORECASE),
    re.compile(r"^\[\s*\d+(?:\s*[,;\-]\s*\d+)*\s*\]"),
    re.compile(r"^(?:doi|isbn|issn)\s*[:=]", re.IGNORECASE),
    re.compile(r"^(?:references?|bibliography|works cited)\b", re.IGNORECASE),
)
_GENERIC_ENTITY_NAMES = {
    "analysis",
    "data",
    "document",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "method",
    "our",
    "paper",
    "research",
    "she",
    "study",
    "that",
    "these",
    "they",
    "this",
    "those",
    "we",
    "work",
    "you",
    "هو",
    "هي",
    "هم",
    "هن",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "هؤلاء",
}


def _valid_entity_name(name: str, category: str) -> bool:
    """Reject obvious extraction artifacts without imposing an entity-count cap."""
    if len(name) < 2 or len(name) > 300 or not category:
        return False
    normalized_name = " ".join(name.casefold().split())
    if (
        normalized_name in _GENERIC_ENTITY_NAMES and not name.isupper()
    ) or len(name.split()) > 20:
        return False
    if any(pattern.search(name) for pattern in _REJECTED_NAME_PATTERNS):
        return False
    sentence_endings = re.search(r"[.!?؟。]\s*$", name)
    if sentence_endings and len(name.split()) > 3:
        return False
    return any(character.isalnum() for character in name)


class EntityExtractor:
    """Extract open-schema graph facts from complete document structure."""

    def __init__(self, client=None) -> None:
        self.client = client or extraction_model_client

    @staticmethod
    def _property(properties: Dict[str, Any], key: str, default: Any = "") -> Any:
        value = properties.get(key, default)
        if isinstance(value, str) and key in {"aliases", "attributes", "mentions"}:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value

    @staticmethod
    def _source_match(
        document: StructuredDocumentInput,
        source_ref: str,
        evidence: str,
    ) -> tuple[Any, Optional[tuple[int, int]]]:
        block, span, _ = EntityExtractor._source_match_details(
            document,
            source_ref,
            evidence,
        )
        return block, span

    @staticmethod
    def _source_match_details(
        document: StructuredDocumentInput,
        source_ref: str,
        evidence: str,
    ) -> tuple[Any, Optional[tuple[int, int]], str]:
        if not evidence.strip():
            return None, None, "missing_evidence"
        normalized_evidence, _, _ = _normalize_evidence_text(evidence)
        if not normalized_evidence:
            return None, None, "missing_evidence"
        candidates = [
            block for block in document.blocks
            if not source_ref or block.source_ref == source_ref
        ]
        if not candidates:
            candidates = document.blocks
        matches = []
        for block in candidates:
            normalized_source, source_starts, source_ends = _normalize_evidence_text(
                block.text
            )
            search_from = 0
            while len(matches) < 2:
                match_start = normalized_source.find(
                    normalized_evidence,
                    search_from,
                )
                if match_start < 0:
                    break
                match_end = match_start + len(normalized_evidence)
                matches.append(
                    (
                        block,
                        (
                            source_starts[match_start],
                            source_ends[match_end - 1],
                        ),
                    )
                )
                search_from = match_start + 1
            if len(matches) > 1:
                break
        if len(matches) == 1:
            return matches[0][0], matches[0][1], ""
        if matches:
            # A precise source_ref already identifies the evidence region. If the
            # same quote occurs twice inside that one region, the first occurrence
            # is sufficient provenance and does not make the fact ambiguous.
            if source_ref and len(candidates) == 1:
                return matches[0][0], matches[0][1], ""
            return None, None, "ambiguous_evidence"

        fuzzy_matches = []
        if len(normalized_evidence) >= 12 and len(normalized_evidence.split()) >= 2:
            expected_tokens = len(normalized_evidence.split())
            for block in candidates:
                normalized_source, source_starts, source_ends = _normalize_evidence_text(
                    block.text
                )
                token_spans = [
                    match.span() for match in re.finditer(r"\S+", normalized_source)
                ]
                for window_size in range(
                    max(2, expected_tokens - 2),
                    expected_tokens + 3,
                ):
                    for start_index in range(0, len(token_spans) - window_size + 1):
                        normalized_start = token_spans[start_index][0]
                        normalized_end = token_spans[start_index + window_size - 1][1]
                        candidate_text = normalized_source[normalized_start:normalized_end]
                        score = SequenceMatcher(
                            None,
                            normalized_evidence,
                            candidate_text,
                            autojunk=False,
                        ).ratio()
                        if score >= 0.86:
                            fuzzy_matches.append(
                                (
                                    score,
                                    block,
                                    (
                                        source_starts[normalized_start],
                                        source_ends[normalized_end - 1],
                                    ),
                                )
                            )
            fuzzy_matches.sort(key=lambda item: item[0], reverse=True)
            if fuzzy_matches:
                best = fuzzy_matches[0]
                second_score = fuzzy_matches[1][0] if len(fuzzy_matches) > 1 else 0.0
                if source_ref or best[0] - second_score >= 0.03:
                    return best[1], best[2], ""
        return None, None, "evidence_not_found"

    @staticmethod
    def _canonical_node_name(
        node_id: Any,
        properties: Dict[str, Any],
        aliases: List[str],
        language: str,
    ) -> tuple[str, List[str]]:
        """Keep the source-language label while retaining every searchable alias."""
        node_name = " ".join(str(node_id or "").split())
        property_name = " ".join(str(properties.get("name") or "").split())
        candidates = [property_name, *aliases, node_name]
        candidates = [value for value in candidates if value]

        def contains_arabic(value: str) -> bool:
            return any("ARABIC" in unicodedata.name(char, "") for char in value)

        name = property_name or node_name
        if language in {"ar", "mixed"}:
            arabic_name = next((value for value in candidates if contains_arabic(value)), "")
            if arabic_name:
                name = arabic_name
        searchable_aliases = list(
            dict.fromkeys(value for value in candidates if value and value != name)
        )
        return name, searchable_aliases

    @staticmethod
    def _source_provenance(
        document: StructuredDocumentInput,
        block: Any,
        span: tuple[int, int],
    ) -> Dict[str, Any]:
        start, end = span
        if block.character_offsets and end > start:
            absolute_start = block.character_offsets[start]
            absolute_end = block.character_offsets[min(end - 1, len(block.character_offsets) - 1)] + 1
        else:
            absolute_start = (
                block.start_offset + start
                if block.start_offset is not None
                else start
            )
            absolute_end = (
                block.start_offset + end
                if block.start_offset is not None
                else end
            )
        return {
            "document_id": document.document_id,
            "page_number": block.page_number,
            "sheet_name": block.sheet_name,
            "section": block.section,
            "source_offsets": [{"start": absolute_start, "end": absolute_end}],
            "bounding_boxes": block.bounding_boxes,
            "extraction_model": settings.EXTRACTION_MODEL,
            "extraction_version": settings.EXTRACTION_VERSION,
        }

    @staticmethod
    def _model_response_records(
        callback: ModelUsageCallbackHandler,
        document: StructuredDocumentInput,
        rendered: str,
    ) -> List[Dict[str, Any]]:
        return [
            {
                **record,
                "language": document.language,
                "source_refs": [block.source_ref for block in document.blocks],
                "input_characters": len(rendered),
                "configured_context_tokens": settings.EXTRACTION_CONTEXT_TOKENS,
                "configured_output_tokens": settings.EXTRACTION_OUTPUT_TOKENS,
            }
            for record in callback.response_records
        ]

    def extract_document(
        self,
        document: StructuredDocumentInput,
    ) -> GraphExtractionResult:
        """Use LangChain's open-schema graph transformer with enforced JSON mode."""
        if not hasattr(self.client, "as_chat_model"):
            raise RuntimeError("The configured extraction client must expose a LangChain chat model")

        chat_model = self.client.as_chat_model()
        if hasattr(chat_model, "bind"):
            chat_model = chat_model.bind(
                max_tokens=settings.EXTRACTION_OUTPUT_TOKENS,
                # Graph extraction needs complete, valid JSON. Thinking models can
                # otherwise spend the entire output budget on hidden reasoning and
                # truncate the JSON before its closing braces.
                reasoning={"enabled": settings.EXTRACTION_REASONING_ENABLED},
            )
        transformer = LLMGraphTransformer(
            llm=JsonModeChatModel(chat_model),
            allowed_nodes=[],
            allowed_relationships=[],
            prompt=GRAPH_TRANSFORMER_PROMPT,
            strict_mode=True,
            node_properties=[
                "name",
                "document_type",
                "evidence",
                "source_ref",
                "confidence",
                "aliases",
                "attributes",
                "description",
            ],
            relationship_properties=[
                "evidence",
                "source_ref",
                "confidence",
                "description",
            ],
        )
        usage_callback = ModelUsageCallbackHandler(
            role=getattr(self.client, "role", "extraction"),
            model=getattr(self.client, "llm_model", ""),
        )
        rendered = document.render()
        source = LangChainDocument(
            page_content=rendered,
            metadata={"document_id": document.document_id},
        )
        try:
            graph_document = transformer.process_response(
                source,
                config={
                    "max_concurrency": settings.LANGCHAIN_MAX_CONCURRENCY,
                    "run_name": "document-open-schema-graph-extraction",
                    "metadata": {
                        "document_id": document.document_id,
                        "document_type": document.document_type,
                    },
                    "callbacks": [usage_callback],
                },
            )
        except Exception as exc:
            response_records = self._model_response_records(
                usage_callback,
                document,
                rendered,
            )
            logger.error(
                "Document graph extraction failed for %s: %s",
                document.document_id,
                exc,
            )
            raise GraphExtractionError(
                f"Graph extraction failed for document {document.document_id}: {exc}",
                model_responses=response_records,
            ) from exc

        result = GraphExtractionResult(document_type=document.document_type)
        result.model_responses = self._model_response_records(
            usage_callback,
            document,
            rendered,
        )
        result.diagnostics.extend(
            {
                "diagnostic_type": "truncated_model_output",
                "finish_reason": response.get("finish_reason"),
                "request_id": response.get("request_id"),
                "source_refs": response.get("source_refs", []),
                "usage": response.get("usage", {}),
            }
            for response in result.model_responses
            if response.get("truncated") is True
        )
        nodes: Dict[tuple[str, str], Any] = {}
        entity_ids: Dict[tuple[str, str], str] = {}
        topic_ids: Dict[tuple[str, str], str] = {}
        topic_parent_links: List[tuple[str, str, str, float, str, Dict[str, Any]]] = []

        for node in graph_document.nodes:
            properties = {
                re.sub(r"(?<!^)([A-Z])", r"_\1", str(key)).lower(): value
                for key, value in (node.properties or {}).items()
            }
            node_type = _category(node.type)
            if node_type.casefold() == "document":
                result.document_type = str(
                    properties.get("document_type") or document.document_type
                ).strip() or "unknown"
                continue
            aliases = self._property(properties, "aliases", [])
            if not isinstance(aliases, list):
                aliases = [str(aliases)] if aliases else []
            name, aliases = self._canonical_node_name(
                node.id,
                properties,
                [str(value) for value in aliases if str(value).strip()],
                document.language,
            )
            evidence = str(properties.get("evidence") or "").strip()
            source_ref = str(properties.get("source_ref") or "").strip()
            confidence = _confidence(properties.get("confidence"))
            block, span, evidence_reason = self._source_match_details(
                document,
                source_ref,
                evidence,
            )
            rejection_reasons = []
            if not name:
                rejection_reasons.append("missing_name")
            if not _valid_entity_name(name, node_type):
                rejection_reasons.append("invalid_entity_name")
            if not evidence:
                rejection_reasons.append("missing_evidence")
            elif evidence_reason:
                rejection_reasons.append(evidence_reason)
            if confidence < settings.EXTRACTION_MIN_CONFIDENCE:
                rejection_reasons.append("low_confidence")
            if (
                rejection_reasons
                or not block
                or span is None
            ):
                result.diagnostics.append(
                    {
                        "diagnostic_type": "rejected_node",
                        "name": name,
                        "category": node_type,
                        "evidence": evidence,
                        "source_ref": source_ref,
                        "confidence": confidence,
                        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
                    }
                )
                continue

            source_evidence = block.text[span[0]:span[1]]
            provenance = self._source_provenance(document, block, span)
            key = (node.id, node_type.casefold())
            nodes[key] = node
            if node_type.casefold() == "topic":
                result.topics.append(
                    ExtractedTopic(
                        name=name,
                        aliases=[str(value) for value in aliases if str(value).strip()],
                        evidence=source_evidence,
                        confidence=confidence,
                        source_ref=source_ref,
                        provenance=provenance,
                    )
                )
                topic_ids[key] = name
                continue

            attributes = self._property(properties, "attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            reserved = {
                "evidence", "source_ref", "confidence", "aliases",
                "attributes", "mentions", "description", "name",
            }
            attributes.update(
                {
                    str(key): str(value)
                    for key, value in properties.items()
                    if key not in reserved and value not in (None, "")
                }
            )
            entity = ExtractedEntity(
                name=name,
                category=node_type,
                description=str(properties.get("description") or ""),
                aliases=[str(value) for value in aliases if str(value).strip()],
                attributes={str(key): str(value) for key, value in attributes.items()},
                evidence=source_evidence,
                confidence=confidence,
                source_ref=source_ref,
                provenance=provenance,
            )
            result.entities.append(entity)
            entity_ids[key] = name

            mention_rows = self._property(properties, "mentions", [])
            if not isinstance(mention_rows, list):
                mention_rows = []
            if isinstance(mention_rows, list):
                for mention in mention_rows:
                    if not isinstance(mention, dict):
                        continue
                    mention_evidence = str(mention.get("evidence") or "").strip()
                    mention_ref = str(mention.get("source_ref") or source_ref)
                    mention_block, mention_span, mention_reason = self._source_match_details(
                        document,
                        mention_ref,
                        mention_evidence,
                    )
                    mention_confidence = _confidence(mention.get("confidence"))
                    mention_reasons = []
                    if not mention_evidence:
                        mention_reasons.append("missing_evidence")
                    elif mention_reason:
                        mention_reasons.append(mention_reason)
                    if mention_confidence < settings.EXTRACTION_MIN_CONFIDENCE:
                        mention_reasons.append("low_confidence")
                    if (
                        mention_reasons
                        or not mention_block
                        or mention_span is None
                    ):
                        result.diagnostics.append(
                            {
                                "diagnostic_type": "rejected_mention",
                                "text": str(mention.get("text") or "").strip(),
                                "entity_name": name,
                                "evidence": mention_evidence,
                                "source_ref": mention_ref,
                                "confidence": mention_confidence,
                                "rejection_reasons": list(
                                    dict.fromkeys(mention_reasons)
                                ),
                            }
                        )
                        continue
                    mention_source_evidence = mention_block.text[
                        mention_span[0]:mention_span[1]
                    ]
                    result.mentions.append(
                        ExtractedMention(
                            text=str(mention.get("text") or "").strip(),
                            entity_name=name,
                            entity_category=node_type,
                            entity_id=name,
                            mention_type=str(mention.get("mention_type") or "name"),
                            evidence=mention_source_evidence,
                            confidence=mention_confidence,
                            source_ref=mention_ref,
                            provenance=self._source_provenance(
                                document,
                                mention_block,
                                mention_span,
                            ),
                        )
                    )

        for relationship in graph_document.relationships:
            source_key = (relationship.source.id, relationship.source.type.casefold())
            target_key = (relationship.target.id, relationship.target.type.casefold())
            properties = {
                re.sub(r"(?<!^)([A-Z])", r"_\1", str(key)).lower(): value
                for key, value in (relationship.properties or {}).items()
            }
            evidence = str(properties.get("evidence") or "").strip()
            source_ref = str(properties.get("source_ref") or "").strip()
            confidence = _confidence(properties.get("confidence"))
            block, span, evidence_reason = self._source_match_details(
                document,
                source_ref,
                evidence,
            )
            rejection_reasons = []
            if source_key not in nodes:
                rejection_reasons.append("missing_source_endpoint")
            if target_key not in nodes:
                rejection_reasons.append("missing_target_endpoint")
            if not _relation_type(relationship.type):
                rejection_reasons.append("invalid_relationship_type")
            if not evidence:
                rejection_reasons.append("missing_evidence")
            elif evidence_reason:
                rejection_reasons.append(evidence_reason)
            if confidence < settings.EXTRACTION_MIN_CONFIDENCE:
                rejection_reasons.append("low_confidence")
            if (
                rejection_reasons
                or not block
                or span is None
            ):
                result.diagnostics.append(
                    {
                        "diagnostic_type": "rejected_relationship",
                        "source": relationship.source.id,
                        "target": relationship.target.id,
                        "relation_type": relationship.type,
                        "evidence": evidence,
                        "source_ref": source_ref,
                        "confidence": confidence,
                        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
                    }
                )
                continue
            source_evidence = block.text[span[0]:span[1]]
            source_is_topic = source_key in topic_ids
            target_is_topic = target_key in topic_ids
            provenance = self._source_provenance(document, block, span)
            if (
                source_is_topic
                and target_is_topic
                and relationship.type.casefold() in {"parent_of", "has_subtopic", "broader_than"}
            ):
                topic_parent_links.append(
                    (
                        topic_ids[target_key],
                        topic_ids[source_key],
                        source_ref,
                        confidence,
                        source_evidence,
                        provenance,
                    )
                )
                continue
            result.relations.append(
                ExtractedRelation(
                    source=str(relationship.source.id),
                    target=str(relationship.target.id),
                    source_category=_category(relationship.source.type),
                    target_category=_category(relationship.target.type),
                    source_entity_id=entity_ids.get(source_key, ""),
                    target_entity_id=entity_ids.get(target_key, ""),
                    relation_type=_relation_type(relationship.type),
                    description=str(properties.get("description") or ""),
                    evidence=source_evidence,
                    confidence=confidence,
                    source_ref=source_ref,
                    provenance=provenance,
                )
            )

        for child_name, parent_name, source_ref, confidence, evidence, provenance in topic_parent_links:
            child = next((topic for topic in result.topics if topic.name == child_name), None)
            if child and parent_name not in child.parent_names:
                child.parent_names.append(parent_name)
        return result

    def extract(
        self,
        text: str,
        document_title: Optional[str] = None,
        chunk_id: str = "",
        known_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphExtractionResult:
        try:
            if hasattr(self.client, "as_chat_model"):
                return self._extract_structured(
                    text=text,
                    document_title=document_title,
                    chunk_id=chunk_id,
                    known_entities=known_entities,
                )

            prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                document_title=document_title or "Unknown paper",
                chunk_id=chunk_id or "unknown",
                known_entities=json.dumps(known_entities or [], ensure_ascii=False),
                text=text[:settings.GRAPH_EXTRACTION_MAX_CHARS],
            )
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt="Return valid JSON only. Preserve evidence and entity IDs exactly.",
                temperature=0.0,
            )
            return self._parse_json_response(
                raw_response,
                chunk_id=chunk_id,
                source_text=text,
            )
        except Exception as exc:
            logger.error("Entity extraction failed for %s: %s", chunk_id, exc)
            raise RuntimeError(f"Graph extraction failed for {chunk_id}: {exc}") from exc

    def _extract_structured(
        self,
        *,
        text: str,
        document_title: Optional[str],
        chunk_id: str,
        known_entities: Optional[List[Dict[str, Any]]],
    ) -> GraphExtractionResult:
        prompt = ChatPromptTemplate.from_messages(
            [("human", EXTRACTION_PROMPT_TEMPLATE)]
        )
        model = self.client.as_chat_model().bind(temperature=0.0)
        json_model = model.bind(response_format={"type": "json_object"})
        usage_callback = ModelUsageCallbackHandler(
            role=getattr(self.client, "role", "extraction"),
            model=getattr(self.client, "llm_model", ""),
        )
        parse_result = RunnableLambda(
            lambda output: self._parse_structured_output(
                output,
                chunk_id=chunk_id,
                source_text=text,
            )
        )
        chain = prompt | json_model | parse_result
        return chain.invoke(
            {
                "document_title": document_title or "Unknown paper",
                "chunk_id": chunk_id or "unknown",
                "known_entities": json.dumps(
                    known_entities or [],
                    ensure_ascii=False,
                ),
                "text": text[:settings.GRAPH_EXTRACTION_MAX_CHARS],
            },
            config={
                "max_concurrency": settings.LANGCHAIN_MAX_CONCURRENCY,
                "run_name": "open-schema-graph-extraction",
                "metadata": {"chunk_id": chunk_id},
                "callbacks": [usage_callback],
            },
        )

    def _parse_structured_output(
        self,
        output: Any,
        *,
        chunk_id: str,
        source_text: str,
    ) -> GraphExtractionResult:
        if isinstance(output, AIMessage):
            tool_calls = output.tool_calls or output.additional_kwargs.get(
                "tool_calls", []
            )
            if tool_calls:
                tool_call = tool_calls[0]
                payload = tool_call.get("args")
                if payload is None:
                    payload = tool_call.get("function", {}).get("arguments")
                if isinstance(payload, str):
                    payload = self._decode_structured_json(payload)
            else:
                content = output.content
                if isinstance(content, list):
                    content = "\n".join(
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and isinstance(block.get("text"), str)
                    )
                payload = self._decode_structured_json(content)
        elif isinstance(output, BaseModel):
            payload = output.model_dump()
        elif isinstance(output, dict):
            payload = output.get("parsed", output)
            if isinstance(payload, BaseModel):
                payload = payload.model_dump()
        elif isinstance(output, str):
            payload = self._decode_structured_json(output)
        else:
            raise TypeError(
                "Structured graph extraction returned unsupported output type "
                f"{type(output).__name__}"
            )
        if not isinstance(payload, dict):
            raise TypeError("Structured graph extraction did not return a JSON object")
        expected_fields = {"entities", "mentions", "relations", "topics"}
        if not expected_fields.intersection(payload):
            raise ValueError(
                "Structured graph extraction returned an object without graph fields"
            )
        try:
            payload = GraphExtractionOutput.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise ValueError(
                "Structured graph extraction did not match the expected schema"
            ) from exc
        return self._parse_json_response(
            json.dumps(payload, ensure_ascii=False),
            chunk_id=chunk_id,
            source_text=source_text,
        )

    @staticmethod
    def _decode_structured_json(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str):
            raise TypeError("Structured graph extraction response was not text")
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("Structured graph extraction response contained no JSON")
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Structured graph extraction response contained invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("Structured graph extraction response was not an object")
        return payload

    def _parse_json_response(
        self,
        raw: str,
        chunk_id: str = "",
        source_text: Optional[str] = None,
    ) -> GraphExtractionResult:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return GraphExtractionResult()
        try:
            data = json.loads(json_match.group(0))
        except (TypeError, json.JSONDecodeError):
            return GraphExtractionResult()

        entities: List[ExtractedEntity] = []
        for row in data.get("entities", []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            category = _category(row.get("category"))
            evidence = str(row.get("evidence") or "").strip()
            confidence = _confidence(row.get("confidence"))
            if (
                not _valid_entity_name(name, category)
                or not _grounded(evidence, source_text)
                or confidence < settings.EXTRACTION_MIN_CONFIDENCE
            ):
                logger.info("Rejected low-quality entity candidate: %r", name)
                continue
            aliases = [str(alias).strip() for alias in row.get("aliases", []) if str(alias).strip()]
            raw_attributes = row.get("attributes") or {}
            attributes = (
                {str(k): str(v) for k, v in raw_attributes.items() if v not in (None, "")}
                if isinstance(raw_attributes, dict) else {}
            )
            entities.append(ExtractedEntity(
                entity_id=str(row.get("entity_id") or "").strip(),
                name=name,
                category=category,
                description=str(row.get("description") or "").strip(),
                aliases=aliases,
                attributes=attributes,
                evidence=evidence,
                confidence=confidence,
                source_ref=str(row.get("source_ref") or "").strip(),
            ))

        mentions: List[ExtractedMention] = []
        for row in data.get("mentions", []):
            if not isinstance(row, dict):
                continue
            mention_text = str(row.get("text") or "").strip()
            evidence = str(row.get("evidence") or "").strip()
            if not mention_text or not _grounded(evidence, source_text):
                continue
            mentions.append(ExtractedMention(
                text=mention_text,
                entity_id=str(row.get("entity_id") or "").strip(),
                entity_name=str(row.get("entity_name") or "").strip(),
                entity_category=_category(row.get("entity_category")),
                mention_type=str(row.get("mention_type") or "name").strip().casefold(),
                attributes={
                    str(key): str(value)
                    for key, value in (row.get("attributes") or {}).items()
                    if value not in (None, "")
                } if isinstance(row.get("attributes") or {}, dict) else {},
                evidence=evidence,
                confidence=_confidence(row.get("confidence")),
                chunk_id=chunk_id,
                source_ref=str(row.get("source_ref") or "").strip(),
            ))

        topics: List[ExtractedTopic] = []
        for row in data.get("topics", []):
            if not isinstance(row, dict):
                continue
            name = " ".join(str(row.get("name") or "").split())
            evidence = str(row.get("evidence") or "").strip()
            if (
                len(name) < 3
                or len(name) > 120
                or not _grounded(evidence, source_text)
                or _confidence(row.get("confidence")) < settings.TOPIC_MIN_CONFIDENCE
                or re.search(r"[.!?؟。]\s*$", name)
            ):
                continue
            raw_parents = row.get("parent_names") or []
            raw_aliases = row.get("aliases") or []
            if not isinstance(raw_parents, list):
                raw_parents = []
            if not isinstance(raw_aliases, list):
                raw_aliases = []
            parent_names = [
                " ".join(str(value).split())
                for value in raw_parents
                if str(value).strip()
            ]
            aliases = [
                " ".join(str(value).split())
                for value in raw_aliases
                if str(value).strip()
            ]
            topics.append(ExtractedTopic(
                name=name,
                parent_names=parent_names,
                aliases=aliases,
                evidence=evidence,
                confidence=_confidence(row.get("confidence")),
                source_ref=str(row.get("source_ref") or "").strip(),
            ))

        relations: List[ExtractedRelation] = []
        for row in data.get("relations", []):
            if not isinstance(row, dict):
                continue
            source = str(row.get("source") or "").strip()
            target = str(row.get("target") or "").strip()
            relation_type = _relation_type(row.get("relation_type"))
            evidence = str(row.get("evidence") or "").strip()
            if (
                not source
                or not target
                or source == target
                or not relation_type
                or not _grounded(evidence, source_text)
                or _confidence(row.get("confidence")) < settings.EXTRACTION_MIN_CONFIDENCE
            ):
                continue
            relations.append(ExtractedRelation(
                source=source,
                target=target,
                source_category=_category(row.get("source_category")),
                target_category=_category(row.get("target_category")),
                source_entity_id=str(row.get("source_entity_id") or "").strip(),
                target_entity_id=str(row.get("target_entity_id") or "").strip(),
                relation_type=relation_type,
                description=str(row.get("description") or "").strip(),
                evidence=evidence,
                confidence=_confidence(row.get("confidence")),
                chunk_id=chunk_id,
                source_ref=str(row.get("source_ref") or "").strip(),
            ))

        return GraphExtractionResult(
            entities=entities,
            mentions=mentions,
            relations=relations,
            topics=topics,
            document_type=str(data.get("document_type") or "unknown"),
        )

    def _rule_based_fallback(self, text: str, chunk_id: str = "") -> GraphExtractionResult:
        """Fail closed: an unavailable model must not pollute the graph."""
        return GraphExtractionResult()
entity_extractor = EntityExtractor()
