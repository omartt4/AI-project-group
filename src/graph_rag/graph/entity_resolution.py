"""Paper-level entity registry for conservative coreference resolution."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from graph_rag.graph.entity_extractor import ExtractedEntity
from graph_rag.graph.topic_taxonomy import topic_taxonomy


GLOBAL_NAME_CATEGORIES = {"Method", "Dataset", "Topic", "Metric", "Task", "Tool"}

_TRAILING_ALIAS = re.compile(r"^(?P<name>.+?)\s*\((?P<alias>[A-Za-z][A-Za-z0-9-]{1,11})\)\s*$")


def normalize_entity_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
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


class PaperEntityRegistry:
    """Keeps stable IDs and known aliases while processing one paper in order."""

    def __init__(self, paper_id: str) -> None:
        self.paper_id = paper_id
        self.entities: Dict[str, ExtractedEntity] = {}
        self._lookup: Dict[Tuple[str, str], List[str]] = {}

    def _new_id(self, entity: ExtractedEntity) -> str:
        category = entity.category.strip().title()
        name = normalize_entity_name(entity.name)
        if category == "Topic":
            assignment = topic_taxonomy.identity_for_entity(entity.name)
            return f"topic:{assignment.key}"
        scope = "global" if category in GLOBAL_NAME_CATEGORIES else self.paper_id
        disambiguator = ""
        if category not in GLOBAL_NAME_CATEGORIES:
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

        key = (normalize_entity_name(entity.name), entity.category.casefold())
        candidates = self._lookup.get(key, [])
        if len(candidates) == 1 and self._compatible(self.entities[candidates[0]], entity):
            current = self.entities[candidates[0]]
            self._merge(current, entity)
            return current

        entity.entity_id = entity.entity_id or self._new_id(entity)
        if entity.entity_id in self.entities:
            self._merge(self.entities[entity.entity_id], entity)
            return self.entities[entity.entity_id]

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
        if current.category in GLOBAL_NAME_CATEGORIES:
            return True
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

