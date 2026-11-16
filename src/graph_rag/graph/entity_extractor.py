"""Evidence-first entity, mention, and relationship extraction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.llm.model_client import extraction_model_client

logger = logging.getLogger(__name__)

ALLOWED_ENTITY_CATEGORIES = {
    "Person", "Organization", "Topic", "Method", "Dataset", "Finding",
    "Metric", "Task", "Tool",
}

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

Allowed entity categories:
Person, Organization, Topic, Method, Dataset, Finding, Metric, Task, Tool

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
    return "Topic"


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
            text=text[:6000],
        )
        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt="Return valid JSON only. Preserve evidence and entity IDs exactly.",
                temperature=0.0,
            )
            return self._parse_json_response(raw_response, chunk_id=chunk_id)
        except Exception as exc:
            logger.warning(
                "Entity extraction failed for chunk %s: %s. Using conservative fallback.",
                chunk_id, exc,
            )
            return self._rule_based_fallback(text, chunk_id=chunk_id)

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
            if len(name) < 2:
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
                category=_category(row.get("category")),
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
        """Extract only explicit acronyms; never invent people or relationships."""
        entities: List[ExtractedEntity] = []
        mentions: List[ExtractedMention] = []
        ignored = {"THE", "FOR", "AND", "WITH", "THAT", "THIS"}
        for acronym in sorted(set(re.findall(r"\b[A-Z]{2,8}\b", text)) - ignored):
            entities.append(ExtractedEntity(
                name=acronym,
                category="Method",
                description="Explicit acronym found by fallback extraction",
            ))
            mentions.append(ExtractedMention(
                text=acronym,
                entity_name=acronym,
                entity_category="Method",
                mention_type="name",
                evidence=acronym,
                confidence=0.35,
                chunk_id=chunk_id,
            ))
        return GraphExtractionResult(entities=entities, mentions=mentions)
entity_extractor = EntityExtractor()
