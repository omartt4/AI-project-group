"""
Entity Extractor module.

Uses local Ollama LLM (qwen2.5) to extract structured entities and relationships
from research paper text chunks.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity discovered in text."""
    name: str
    category: str  # 'Author', 'Topic', 'Method', 'Dataset', 'Finding', 'Tool'
    description: str = ""


@dataclass
class ExtractedRelation:
    """A semantic relation between two entities."""
    source: str
    target: str
    relation_type: str  # 'AUTHORED', 'PROPOSES', 'EVALUATES_ON', 'USES_METHOD', 'PART_OF_TOPIC'
    description: str = ""


@dataclass
class GraphExtractionResult:
    """Entities and relationships extracted from a chunk."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)


EXTRACTION_PROMPT_TEMPLATE = """You are an expert knowledge graph engineer analyzing a scientific research paper.
Extract the key scientific entities and relationships from the text below.

Entity categories allowed:
- 'Topic': research area or subfield (e.g. "Speaker Diarization", "Relation Extraction", "Capsule Networks")
- 'Method': model, technique, architecture, algorithm (e.g. "BERT", "LSTM", "RA-CapNet")
- 'Dataset': benchmark or dataset mentioned (e.g. "CALLHOME", "AMI", "DIHARD II")
- 'Author': researcher name if mentioned
- 'Finding': key insight or conclusion

Relation types allowed:
- 'PROPOSES': (Paper/Author -> Method)
- 'EVALUATES_ON': (Method -> Dataset)
- 'USES_METHOD': (Method -> Method)
- 'PART_OF_TOPIC': (Method/Dataset -> Topic)
- 'RELATED_TO': (Entity -> Entity)

Respond ONLY with valid JSON in this exact structure:
{{
  "entities": [
    {{"name": "Speaker Diarization", "category": "Topic", "description": "Audio segmentation by speaker"}},
    {{"name": "RA-CapNet", "category": "Method", "description": "Regularized Attentive Capsule Network"}}
  ],
  "relations": [
    {{"source": "RA-CapNet", "target": "Relation Extraction", "relation_type": "PART_OF_TOPIC", "description": "Applied to"}}
  ]
}}

Text to analyze:
{text}
"""


class EntityExtractor:
    """Extracts entities and relationships using Ollama."""

    def __init__(self, client=None) -> None:
        self.client = client or ollama_client

    def extract(self, text: str, document_title: Optional[str] = None) -> GraphExtractionResult:
        """Extract entities and relations from text chunk."""
        # Truncate text if very long for LLM prompt
        prompt_text = text[:1500] if len(text) > 1500 else text
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=prompt_text)

        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt="You are a JSON-only extraction engine. Output valid JSON without preamble.",
                temperature=0.1,
            )

            return self._parse_json_response(raw_response, document_title)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}. Falling back to rule-based extraction.")
            return self._rule_based_fallback(text, document_title)

    def _parse_json_response(self, raw: str, document_title: Optional[str]) -> GraphExtractionResult:
        """Parse and validate JSON response from LLM."""
        # Find JSON object boundaries
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return GraphExtractionResult()

        try:
            data = json.loads(json_match.group(0))
        except Exception:
            return GraphExtractionResult()

        entities: List[ExtractedEntity] = []
        for e in data.get("entities", []):
            name = str(e.get("name", "")).strip()
            cat = str(e.get("category", "Topic")).strip()
            desc = str(e.get("description", "")).strip()
            if name and len(name) >= 2:
                entities.append(ExtractedEntity(name=name, category=cat, description=desc))

        relations: List[ExtractedRelation] = []
        for r in data.get("relations", []):
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            rtype = str(r.get("relation_type", "RELATED_TO")).strip().upper()
            desc = str(r.get("description", "")).strip()
            if src and tgt and src != tgt:
                relations.append(
                    ExtractedRelation(
                        source=src,
                        target=tgt,
                        relation_type=rtype,
                        description=desc,
                    )
                )

        return GraphExtractionResult(entities=entities, relations=relations)

    def _rule_based_fallback(self, text: str, document_title: Optional[str]) -> GraphExtractionResult:
        """Lightweight heuristic extraction if LLM is unavailable."""
        entities: List[ExtractedEntity] = []
        relations: List[ExtractedRelation] = []

        if document_title:
            entities.append(ExtractedEntity(name=document_title, category="Method", description="Document Title"))

        # Look for capitalized acronyms (e.g. BERT, CNN, LSTM, ASR, NLP)
        acronyms = set(re.findall(r"\b[A-Z]{2,6}\b", text))
        for acr in acronyms:
            if acr not in {"THE", "FOR", "AND", "WITH", "THAT", "THIS"}:
                entities.append(ExtractedEntity(name=acr, category="Method", description="Model/Acronym"))
                if document_title:
                    relations.append(
                        ExtractedRelation(
                            source=document_title,
                            target=acr,
                            relation_type="USES_METHOD",
                        )
                    )

        return GraphExtractionResult(entities=entities, relations=relations)


entity_extractor = EntityExtractor()
