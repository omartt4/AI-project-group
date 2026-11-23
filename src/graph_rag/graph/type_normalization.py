"""Open-world type normalization using graph vocabulary and semantic similarity."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence

from graph_rag.graph.entity_extractor import GraphExtractionResult

logger = logging.getLogger(__name__)

_GENERIC_SYNONYMS = {
    "company": "Organization",
    "corporation": "Organization",
    "university": "Organization",
    "institution": "Organization",
    "researcher": "Person",
    "author": "Person",
    "algorithm": "Method",
    "technique": "Method",
    "methodology": "Method",
}


def normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return -1.0
    return dot / (left_norm * right_norm)


class TypeNormalizer:
    """Canonicalize labels without restricting valid novel types."""

    def __init__(
        self,
        embedding_client: Any,
        *,
        similarity_threshold: float = 0.94,
        minimum_margin: float = 0.04,
    ) -> None:
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.minimum_margin = minimum_margin
        self._embedding_cache: Dict[str, List[float]] = {}

    def _embeddings(self, labels: Iterable[str]) -> Dict[str, List[float]]:
        unique = list(dict.fromkeys(label for label in labels if label))
        missing = [label for label in unique if normalize_label(label) not in self._embedding_cache]
        if missing:
            if not getattr(self.embedding_client, "embed_model", ""):
                return {}
            vectors = self.embedding_client.get_embeddings_batch(missing)
            for label, vector in zip(missing, vectors):
                self._embedding_cache[normalize_label(label)] = vector
        return {
            normalize_label(label): self._embedding_cache[normalize_label(label)]
            for label in unique
            if normalize_label(label) in self._embedding_cache
        }

    def _normalize(
        self,
        label: str,
        existing_labels: List[str],
        context: str,
        embeddings: Dict[str, List[float]],
        *,
        allow_synonyms: bool = True,
    ) -> str:
        normalized = normalize_label(label)
        if not normalized:
            return ""
        exact = [
            existing
            for existing in existing_labels
            if normalize_label(existing) == normalized
        ]
        if len(exact) == 1:
            return exact[0]
        fallback = _GENERIC_SYNONYMS.get(normalized) if allow_synonyms else None
        if fallback:
            if normalized in {"algorithm", "technique", "methodology"}:
                context_text = context.casefold()
                if not any(term in context_text for term in ("method", "approach", "procedure")):
                    return " ".join(word.capitalize() for word in label.split())
            return fallback

        vector = embeddings.get(normalized)
        if vector is not None and existing_labels:
            scored = sorted(
                (
                    _cosine(vector, embeddings.get(normalize_label(existing), [])),
                    existing,
                )
                for existing in existing_labels
                if embeddings.get(normalize_label(existing)) is not None
            )
            if scored and scored[-1][0] >= self.similarity_threshold and (
                len(scored) == 1
                or scored[-1][0] - scored[-2][0] >= self.minimum_margin
            ):
                return scored[-1][1]
        return " ".join(word.capitalize() for word in label.split())

    def normalize_result(
        self,
        result: GraphExtractionResult,
        *,
        existing_entity_types: List[str],
        existing_relationship_types: List[str],
    ) -> GraphExtractionResult:
        entity_labels = {
            entity.category for entity in result.entities if entity.category
        } | {
            category
            for relation in result.relations
            for category in (relation.source_category, relation.target_category)
            if category
        }
        relationship_labels = {
            relation.relation_type
            for relation in result.relations
            if relation.relation_type
        }
        labels = (
            entity_labels | set(existing_entity_types)
            if existing_entity_types
            else set()
        ) | (
            relationship_labels | set(existing_relationship_types)
            if existing_relationship_types
            else set()
        )
        embeddings: Dict[str, List[float]] = {}
        if self.embedding_client and getattr(self.embedding_client, "embed_model", ""):
            try:
                embeddings = self._embeddings(labels)
            except Exception as exc:
                logger.warning("Semantic type normalization unavailable: %s", exc)
                result.diagnostics.append(
                    {
                        "diagnostic_type": "type_similarity_unavailable",
                        "error": str(exc),
                    }
                )

        for entity in result.entities:
            context = f"{entity.name} {entity.description} " + " ".join(
                f"{key} {value}" for key, value in entity.attributes.items()
            )
            entity.category = self._normalize(
                entity.category,
                existing_entity_types,
                context,
                embeddings,
            )
        for relation in result.relations:
            context = f"{relation.description} {relation.evidence}"
            relation.source_category = self._normalize(
                relation.source_category,
                existing_entity_types,
                context,
                embeddings,
            )
            relation.target_category = self._normalize(
                relation.target_category,
                existing_entity_types,
                context,
                embeddings,
            )
            relation.relation_type = self._normalize(
                relation.relation_type,
                existing_relationship_types,
                context,
                embeddings,
                allow_synonyms=False,
            ).replace(" ", "_").upper()
        return result
