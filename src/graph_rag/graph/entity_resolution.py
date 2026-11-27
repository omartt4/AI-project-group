"""Document-level entity registry for conservative coreference resolution."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

from graph_rag.graph.entity_extractor import ExtractedEntity, ExtractedRelation
from graph_rag.graph.topic_taxonomy import topic_taxonomy


_TRAILING_ALIAS = re.compile(r"^(?P<name>.+?)\s*\((?P<alias>[A-Za-z][A-Za-z0-9-]{1,11})\)\s*$")


def normalize_entity_name(value: str) -> str:
    """Normalize Latin/Arabic name variants without changing stored display text."""
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
    normalized: List[str] = []
    for character in text:
        codepoint = ord(character)
        if (
            character == "\u0640"
            or 0x064B <= codepoint <= 0x065F
            or codepoint == 0x0670
            or 0x06D6 <= codepoint <= 0x06ED
            or 0x08D3 <= codepoint <= 0x08FF
        ):
            continue
        normalized.append(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ى": "ي",
                "ئ": "ي",
                "ؤ": "و",
            }.get(character, character)
        )
    text = "".join(normalized)
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def canonicalize_entity(entity: ExtractedEntity) -> ExtractedEntity:
    """Move a trailing acronym into aliases so common name variants share one ID."""
    match = _TRAILING_ALIAS.match(entity.name.strip())
    if not match:
        return entity
    canonical = match.group("name").strip()
    alias = match.group("alias").strip()
    entity.aliases = sorted(set([*entity.aliases, alias, entity.name]))
    entity.name = canonical
    return entity


class DocumentEntityRegistry:
    """Keeps stable IDs and known aliases while processing one document."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.paper_id = document_id
        self.entities: Dict[str, ExtractedEntity] = {}
        self._lookup: Dict[Tuple[str, str], List[str]] = {}

    def _new_id(self, entity: ExtractedEntity) -> str:
        category = entity.category.strip().title()
        name = normalize_entity_name(entity.name)
        if category == "Topic":
            assignment = topic_taxonomy.identity_for_entity(entity.name)
            return f"topic:{assignment.key}"
        scope = self.document_id
        disambiguator = normalize_entity_name(entity.description)[:80]
        digest = hashlib.sha256(
            f"{scope}|{category.casefold()}|{name}|{disambiguator}".encode("utf-8")
        ).hexdigest()[:24]
        return f"ent-{digest}"

    def register(self, entity: ExtractedEntity) -> ExtractedEntity:
        entity = canonicalize_entity(entity)
        if entity.entity_id and entity.entity_id in self.entities:
            current = self.entities[entity.entity_id]
            self._merge(current, entity)
            return current
        # Only IDs already issued by this registry may be reused by the model.
        # Unknown model-generated IDs are discarded and replaced deterministically.
        entity.entity_id = ""

        candidates = {
            candidate
            for value in [entity.name, *entity.aliases]
            for candidate in self._lookup.get(
                (normalize_entity_name(value), entity.category.casefold()),
                [],
            )
        }
        compatible = [
            candidate
            for candidate in candidates
            if self._compatible(self.entities[candidate], entity)
        ]
        if len(compatible) == 1:
            current = self.entities[compatible[0]]
            self._merge(current, entity)
            return current

        entity.entity_id = entity.entity_id or self._new_id(entity)
        if entity.entity_id in self.entities:
            self._merge(self.entities[entity.entity_id], entity)
            return self.entities[entity.entity_id]

        if entity.evidence:
            entity.provenance = dict(entity.provenance or {})
            entity.provenance.setdefault(
                "evidence_records",
                [
                    {
                        "evidence": entity.evidence,
                        "confidence": entity.confidence,
                        "source_ref": entity.source_ref,
                        **entity.provenance,
                    }
                ],
            )
        self.entities[entity.entity_id] = entity
        self._index(entity)
        return entity

    def seed(self, entity: ExtractedEntity) -> ExtractedEntity:
        """Trust one canonical entity loaded from the existing graph."""
        if not entity.entity_id:
            return self.register(entity)
        current = self.entities.get(entity.entity_id)
        if current:
            self._merge(current, entity)
            return current
        self.entities[entity.entity_id] = entity
        self._index(entity)
        return entity

    @staticmethod
    def _compatible(current: ExtractedEntity, incoming: ExtractedEntity) -> bool:
        """Return whether a name match has enough evidence to merge safely."""
        shared_keys = set(current.attributes) & set(incoming.attributes)
        for key in shared_keys:
            left = normalize_entity_name(current.attributes[key])
            right = normalize_entity_name(incoming.attributes[key])
            if left and right and left != right:
                return False
        if current.description and incoming.description:
            left = normalize_entity_name(current.description)
            right = normalize_entity_name(incoming.description)
            if left != right and not shared_keys:
                return False
        return True

    def _index(self, entity: ExtractedEntity) -> None:
        names = [entity.name, *entity.aliases]
        for value in names:
            key = (normalize_entity_name(value), entity.category.casefold())
            if not key[0]:
                continue
            ids = self._lookup.setdefault(key, [])
            if entity.entity_id not in ids:
                ids.append(entity.entity_id)

    def _merge(self, current: ExtractedEntity, incoming: ExtractedEntity) -> None:
        if len(incoming.description) > len(current.description):
            current.description = incoming.description
        current.aliases = sorted(set(current.aliases) | set(incoming.aliases))
        current.attributes.update({k: v for k, v in incoming.attributes.items() if v})
        evidence_records = list(
            current.provenance.get("evidence_records", [])
            if isinstance(current.provenance, dict)
            else []
        )
        current_provenance = {
            key: value
            for key, value in (current.provenance or {}).items()
            if key != "evidence_records"
        }
        current_record = {
            "evidence": current.evidence,
            "confidence": current.confidence,
            "source_ref": current.source_ref,
            **current_provenance,
        }
        if current.evidence and current_record not in evidence_records:
            evidence_records.append(current_record)
        incoming_provenance = {
            key: value
            for key, value in (incoming.provenance or {}).items()
            if key != "evidence_records"
        }
        incoming_record = {
            "evidence": incoming.evidence,
            "confidence": incoming.confidence,
            "source_ref": incoming.source_ref,
            **incoming_provenance,
        }
        if incoming.evidence and incoming_record not in evidence_records:
            evidence_records.append(incoming_record)
        if incoming.confidence > current.confidence:
            current.evidence = incoming.evidence
            current.confidence = incoming.confidence
            current.source_ref = incoming.source_ref
            current.provenance = incoming_provenance
        current.provenance = {
            key: value
            for key, value in (current.provenance or {}).items()
            if key != "evidence_records"
        }
        if evidence_records:
            current.provenance["evidence_records"] = evidence_records
        self._index(current)

    def resolve(
        self,
        entity_id: str = "",
        name: str = "",
        category: str = "",
    ) -> Optional[str]:
        if entity_id and entity_id in self.entities:
            return entity_id
        normalized = normalize_entity_name(name)
        if not normalized:
            return None
        if category:
            candidates = self._lookup.get((normalized, category.casefold()), [])
        else:
            candidates = [
                candidate
                for (known_name, _), ids in self._lookup.items()
                if known_name == normalized
                for candidate in ids
            ]
        unique = list(dict.fromkeys(candidates))
        return unique[0] if len(unique) == 1 else None

    def known_entities(self) -> List[dict]:
        return [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "category": entity.category,
                "aliases": entity.aliases,
                "description": entity.description[:240],
                "attributes": entity.attributes,
            }
            for entity in self.entities.values()
        ]


# Backward-compatible name for existing callers and cached integrations.
PaperEntityRegistry = DocumentEntityRegistry


class EntityResolver:
    """Resolve document-local entities against canonical cross-document entities."""

    def __init__(
        self,
        graph: Any,
        vector_store: Any,
        *,
        similarity_threshold: float = 0.92,
        minimum_margin: float = 0.05,
    ) -> None:
        self.graph = graph
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.minimum_margin = minimum_margin

    @staticmethod
    def _category_matches(left: str, right: str) -> bool:
        return normalize_entity_name(left) == normalize_entity_name(right)

    @staticmethod
    def _attributes_match(entity: ExtractedEntity, candidate: Dict[str, Any]) -> bool:
        attributes = candidate.get("attributes") or candidate.get("attributes_json") or {}
        if isinstance(attributes, str):
            try:
                attributes = json.loads(attributes)
            except json.JSONDecodeError:
                attributes = {}
        if not isinstance(attributes, dict):
            return True
        for key in set(entity.attributes) & set(attributes):
            incoming = normalize_entity_name(str(entity.attributes[key]))
            existing = normalize_entity_name(str(attributes[key]))
            if incoming and existing and incoming != existing:
                return False
        return True

    def resolve(
        self,
        entities: List[ExtractedEntity],
        relations: Optional[List[ExtractedRelation]] = None,
    ) -> tuple[List[ExtractedEntity], List[Dict[str, Any]], set[str], Dict[str, str]]:
        resolved: List[ExtractedEntity] = []
        diagnostics: List[Dict[str, Any]] = []
        ambiguous_ids: set[str] = set()
        id_map: Dict[str, str] = {}
        unresolved: List[ExtractedEntity] = []
        queries: List[str] = []
        relationship_context = self._relationship_context(entities, relations or [])
        for entity in entities:
            original_id = entity.entity_id
            contexts = relationship_context.get(entity.entity_id, [])
            lookup_kwargs = {
                "name": entity.name,
                "category": entity.category,
                "aliases": entity.aliases,
                "attributes": entity.attributes,
            }
            if contexts:
                lookup_kwargs["relationship_context"] = contexts
            exact = self.graph.find_entity_candidates(**lookup_kwargs)
            exact_by_id = {
                str(candidate.get("entity_id") or ""): candidate
                for candidate in exact
                if candidate.get("entity_id")
                and self._category_matches(
                    str(candidate.get("category") or ""),
                    entity.category,
                )
                and self._attributes_match(entity, candidate)
            }
            if len(exact_by_id) == 1:
                candidate = next(iter(exact_by_id.values()))
                self._merge_candidate(entity, candidate)
                id_map[original_id] = entity.entity_id
                resolved.append(entity)
                continue
            if len(exact_by_id) > 1:
                diagnostics.append(
                    self._ambiguity(entity, list(exact_by_id.values()), "exact_identity")
                )
                ambiguous_ids.add(entity.entity_id)
                id_map[original_id] = ""
                continue
            unresolved.append(entity)
            queries.append(
                " ".join(
                    [
                        entity.name,
                        entity.category,
                        entity.description,
                        *entity.aliases,
                        *[
                            f"{key} {value}"
                            for key, value in entity.attributes.items()
                        ],
                        *[
                            " ".join(
                                filter(
                                    None,
                                    [
                                        context.get("relation_type", ""),
                                        context.get("other_name", ""),
                                        context.get("other_category", ""),
                                    ],
                                )
                            )
                            for context in contexts
                        ],
                    ]
                ).strip()
            )

        semantic_results = self.vector_store.search_entity_vectors_batch(queries)
        for entity, candidates in zip(unresolved, semantic_results):
            original_id = entity.entity_id
            semantic = [
                candidate
                for candidate in candidates
                if self._category_matches(
                    str(candidate.get("category") or ""),
                    entity.category,
                )
                and self._attributes_match(entity, candidate)
            ]
            eligible = [
                candidate
                for candidate in semantic
                if float(candidate.get("similarity_score", 0.0))
                >= self.similarity_threshold
            ]
            eligible.sort(
                key=lambda candidate: float(candidate.get("similarity_score", 0.0)),
                reverse=True,
            )
            if eligible and (
                len(eligible) == 1
                or float(eligible[0]["similarity_score"])
                - float(eligible[1]["similarity_score"])
                >= self.minimum_margin
            ):
                self._merge_candidate(entity, eligible[0])
                id_map[original_id] = entity.entity_id
                resolved.append(entity)
                continue
            if eligible:
                diagnostics.append(self._ambiguity(entity, eligible, "semantic_similarity"))
                ambiguous_ids.add(entity.entity_id)
                id_map[original_id] = ""
                continue
            id_map[original_id] = entity.entity_id
            resolved.append(entity)
        return resolved, diagnostics, ambiguous_ids, id_map

    @staticmethod
    def _relationship_context(
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
    ) -> Dict[str, List[Dict[str, str]]]:
        """Describe each entity by its local semantic neighbours for resolution."""
        by_id = {entity.entity_id: entity for entity in entities if entity.entity_id}
        by_name = {
            (normalize_entity_name(entity.name), entity.category.casefold()): entity
            for entity in entities
        }

        def endpoint(
            entity_id: str,
            name: str,
            category: str,
        ) -> Optional[ExtractedEntity]:
            return by_id.get(entity_id) or by_name.get(
                (normalize_entity_name(name), category.casefold())
            )

        context: Dict[str, List[Dict[str, str]]] = {}
        for relation in relations:
            source = endpoint(
                relation.source_entity_id,
                relation.source,
                relation.source_category,
            )
            target = endpoint(
                relation.target_entity_id,
                relation.target,
                relation.target_category,
            )
            if source and target:
                context.setdefault(source.entity_id, []).append(
                    {
                        "relation_type": relation.relation_type,
                        "other_name": target.name,
                        "other_category": target.category,
                    }
                )
                context.setdefault(target.entity_id, []).append(
                    {
                        "relation_type": relation.relation_type,
                        "other_name": source.name,
                        "other_category": source.category,
                    }
                )
        return context

    @staticmethod
    def _merge_candidate(
        entity: ExtractedEntity,
        candidate: Dict[str, Any],
    ) -> None:
        original_name = entity.name
        entity.entity_id = str(candidate["entity_id"])
        entity.name = str(candidate.get("name") or entity.name)
        aliases = candidate.get("aliases") or candidate.get("aliases_json") or []
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except json.JSONDecodeError:
                aliases = [aliases]
        entity.aliases = sorted(set(entity.aliases + list(aliases) + [original_name]))
        try:
            attributes = candidate.get("attributes") or candidate.get("attributes_json") or {}
            if isinstance(attributes, str):
                attributes = json.loads(attributes)
        except json.JSONDecodeError:
            attributes = {}
        if isinstance(attributes, dict):
            entity.attributes = {
                **attributes,
                **{key: value for key, value in entity.attributes.items() if value},
            }

    @staticmethod
    def _ambiguity(
        entity: ExtractedEntity,
        candidates: List[Dict[str, Any]],
        resolution_method: str,
    ) -> Dict[str, Any]:
        return {
            "diagnostic_type": "ambiguous_entity_resolution",
            "name": entity.name,
            "category": entity.category,
            "evidence": entity.evidence,
            "confidence": entity.confidence,
            "resolution_method": resolution_method,
            "candidates": [
                {
                    "entity_id": candidate.get("entity_id", ""),
                    "name": candidate.get("name", ""),
                    "category": candidate.get("category", ""),
                    "similarity_score": candidate.get("similarity_score"),
                }
                for candidate in candidates
            ],
        }

