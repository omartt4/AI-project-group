"""Evidence-first entity, mention, and relationship extraction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.config import settings
from graph_rag.llm.model_client import extraction_model_client

logger = logging.getLogger(__name__)

ALLOWED_ENTITY_CATEGORIES = frozenset({
    "Person", "Organization", "Method", "Dataset",
})

ALLOWED_RELATION_TYPES = {
    "AUTHORED", "AFFILIATED_WITH", "HOLDS_ROLE", "PROPOSES",
    "EVALUATES_ON", "USES_METHOD", "PART_OF_TOPIC", "REPORTS", "RELATED_TO",
}


@dataclass
class ExtractedEntity:
    """Canonical entity candidate within the current paper."""

    name: str
    category: str
    description: str = ""
    entity_id: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedMention:
    """A source phrase that refers to an entity."""

    text: str
    entity_name: str = ""
    entity_category: str = ""
    entity_id: str = ""
    mention_type: str = "name"
    evidence: str = ""
    confidence: float = 0.0
    chunk_id: str = ""
    mention_id: str = ""


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


@dataclass
class GraphExtractionResult:
    """Entities, their textual mentions, and evidence-backed relations."""

    entities: List[ExtractedEntity] = field(default_factory=list)
    mentions: List[ExtractedMention] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)


EXTRACTION_PROMPT_TEMPLATE = """You extract an evidence-first knowledge graph from one research-paper chunk.

Document title: {document_title}
Chunk ID: {chunk_id}

Entities already known in this paper:
{known_entities}

Rules:
1. Reuse an existing entity_id when a name, alias, pronoun, or role phrase refers to a known entity.
2. A mention is the exact phrase in this chunk. An entity is the real object it refers to.
3. Do not create a new Person for phrases such as "the director", "the doctor", "he", or "she" unless the text clearly introduces a different person.
4. If a mention is ambiguous, keep entity_id empty and lower its confidence.
5. Every relation must quote short supporting evidence from this chunk.
6. Do not infer facts that the text does not state.
7. New entities must leave entity_id empty; the application assigns their stable ID.
8. Do not extract Topic entities here. Paper topics are classified once from title,
   abstract, keywords, and metadata by the topic-taxonomy stage.
9. Create entities only for reusable, named objects. Do not create entities for
   table/figure references, citation groups such as "Zhu et al.", parameter values,
   generic words such as "accuracy", or extraction commentary.
10. Findings, metrics, tasks, and tools are not nodes. Preserve them in a
    relation description, evidence, or entity attributes when relevant.
11. Extract only methods and datasets actually proposed, used, or evaluated by
    this paper; ignore items merely listed in related work or references.
12. Do not create Person nodes solely from the paper byline or bibliography.
13. Put acronyms and alternate spellings in aliases instead of creating another entity.
14. For a Person identified by a role phrase, set attributes.role and
    attributes.organization whenever the text states them.

Allowed entity categories:
Person, Organization, Method, Dataset

Allowed relation types:
AUTHORED, AFFILIATED_WITH, HOLDS_ROLE, PROPOSES, EVALUATES_ON,
USES_METHOD, PART_OF_TOPIC, REPORTS, RELATED_TO

Return JSON only:
{{
  "entities": [{{"entity_id": "existing ID or empty", "name": "canonical name", "category": "allowed category", "description": "disambiguating description", "aliases": ["alternate name"], "attributes": {{"role": "CEO", "organization": "Example Company"}}}}],
  "mentions": [{{"text": "the director", "entity_id": "known ID when resolved", "entity_name": "canonical name", "entity_category": "Person", "mention_type": "name|alias|pronoun|role", "evidence": "source sentence", "confidence": 0.0}}],
  "relations": [{{"source_entity_id": "known ID when available", "source": "canonical source name", "source_category": "allowed category", "target_entity_id": "known ID when available", "target": "canonical target name", "target_category": "allowed category", "relation_type": "allowed relation", "description": "short explanation", "evidence": "source sentence", "confidence": 0.0}}]
}}

Chunk text:
{text}
"""


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _category(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    for allowed in ALLOWED_ENTITY_CATEGORIES:
        if allowed.casefold() == raw:
            return allowed
    return ""


_REJECTED_NAME_PATTERNS = (
    re.compile(r"^\d+\s+", re.IGNORECASE),
    re.compile(r"^(?:table|figure|equation|section)\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:name garbled|extraction error|cited work)\b", re.IGNORECASE),
    re.compile(r"\bet\s+al\.?$", re.IGNORECASE),
)

_GENERIC_ENTITY_NAMES = {
    "accuracy", "precision", "recall", "inference time",
    "trainable parameters", "objective evaluation scores",
}


def _valid_entity_name(name: str, category: str) -> bool:
    """Reject obvious extraction artifacts without imposing an entity-count cap."""
    if len(name) < 2 or not category:
        return False
    if any(pattern.search(name) for pattern in _REJECTED_NAME_PATTERNS):
        return False
    if name.casefold().strip() in _GENERIC_ENTITY_NAMES:
        return False
    word_count = len(re.findall(r"\b\w+\b", name, flags=re.UNICODE))
    return word_count <= 12


class EntityExtractor:
    """Extract graph facts while carrying paper-level entity context forward."""

    def __init__(self, client=None) -> None:
        self.client = client or extraction_model_client

    def extract(
        self,
        text: str,
        document_title: Optional[str] = None,
        chunk_id: str = "",
        known_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphExtractionResult:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            document_title=document_title or "Unknown paper",
            chunk_id=chunk_id or "unknown",
            known_entities=json.dumps(known_entities or [], ensure_ascii=False),
            text=text[:settings.GRAPH_EXTRACTION_MAX_CHARS],
        )
        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt="Return valid JSON only. Preserve evidence and entity IDs exactly.",
                temperature=0.0,
            )
            return self._parse_json_response(raw_response, chunk_id=chunk_id)
        except Exception as exc:
            logger.error("Entity extraction failed for %s: %s", chunk_id, exc)
            raise RuntimeError(f"Graph extraction failed for {chunk_id}: {exc}") from exc

    def _parse_json_response(self, raw: str, chunk_id: str = "") -> GraphExtractionResult:
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
            if not _valid_entity_name(name, category):
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
            ))

        mentions: List[ExtractedMention] = []
        for row in data.get("mentions", []):
            if not isinstance(row, dict):
                continue
            mention_text = str(row.get("text") or "").strip()
            if not mention_text:
                continue
            mentions.append(ExtractedMention(
                text=mention_text,
                entity_id=str(row.get("entity_id") or "").strip(),
                entity_name=str(row.get("entity_name") or "").strip(),
                entity_category=_category(row.get("entity_category")),
                mention_type=str(row.get("mention_type") or "name").strip().casefold(),
                evidence=str(row.get("evidence") or "").strip(),
                confidence=_confidence(row.get("confidence")),
                chunk_id=chunk_id,
            ))

        relations: List[ExtractedRelation] = []
        for row in data.get("relations", []):
            if not isinstance(row, dict):
                continue
            source = str(row.get("source") or "").strip()
            target = str(row.get("target") or "").strip()
            relation_type = str(row.get("relation_type") or "").strip().upper()
            if not source or not target or source == target:
                continue
            if relation_type not in ALLOWED_RELATION_TYPES:
                logger.warning("Rejected unsupported relation type: %s", relation_type)
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
                evidence=str(row.get("evidence") or "").strip(),
                confidence=_confidence(row.get("confidence")),
                chunk_id=chunk_id,
            ))

        return GraphExtractionResult(entities=entities, mentions=mentions, relations=relations)

    def _rule_based_fallback(self, text: str, chunk_id: str = "") -> GraphExtractionResult:
        """Fail closed: an unavailable model must not pollute the graph."""
        return GraphExtractionResult()
entity_extractor = EntityExtractor()
