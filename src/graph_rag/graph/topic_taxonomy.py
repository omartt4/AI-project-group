"""Hierarchical topic taxonomy with controlled parents and extensible subtopics."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple


def normalize_topic(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


SEED_TOPICS: Dict[str, set[str]] = {
    "Machine Learning": {"machine learning", "ml"},
    "Computer Vision": {"computer vision", "cv", "image processing"},
    "Natural Language Processing": {
        "natural language processing",
        "nlp",
        "computational linguistics",
    },
}

PARENT_HINTS: Dict[str, set[str]] = {
    "Machine Learning": {
        "machine learning", "deep learning", "neural", "classification",
        "clustering", "optimization", "reinforcement learning", "feature extractor",
        "electronic design automation", "integrated circuit", "design for manufacturing",
    },
    "Computer Vision": {
        "computer vision", "image", "visual", "face", "object detection",
        "image segmentation", "generative adversarial", "vision",
    },
    "Natural Language Processing": {
        "natural language", "nlp", "language model", "text", "speech",
        "speaker diarization", "diarization", "relation extraction",
        "question answering", "translation", "sentiment", "named entity",
    },
}


@dataclass(frozen=True)
class TopicAssignment:
    """One top-level topic or subtopic assignment for a paper."""

    name: str
    key: str
    status: str
    source: str
    confidence: float
    level: str = "top_level"
    parent_names: Tuple[str, ...] = ()
    parent_keys: Tuple[str, ...] = ()


class TopicTaxonomy:
    """Build a controlled top-level taxonomy while retaining new subtopics."""

    def __init__(self) -> None:
        self.aliases = {
            normalize_topic(alias): canonical
            for canonical, aliases in SEED_TOPICS.items()
            for alias in aliases | {canonical}
        }

    @staticmethod
    def _key(name: str) -> str:
        return normalize_topic(name).replace(" ", "-") or "unclassified"

    def _top_level(self, name: str, source: str, confidence: float = 1.0) -> TopicAssignment:
        controlled = name in SEED_TOPICS
        return TopicAssignment(
            name=name,
            key=self._key(name),
            status="active" if controlled else "candidate",
            source="controlled_alias" if controlled else source,
            confidence=confidence,
            level="top_level",
        )

    def _hinted_parents(self, value: str) -> Tuple[str, ...]:
        normalized = normalize_topic(value)
        matches = []
        for parent, hints in PARENT_HINTS.items():
            if any(normalize_topic(hint) in normalized for hint in hints):
                matches.append(parent)
        return tuple(matches)

    def assign(self, raw_topics: str, source: str = "metadata") -> List[TopicAssignment]:
        values = [part.strip() for part in re.split(r"[;,|]", raw_topics or "") if part.strip()]
        if not values:
            values = ["Unclassified"]

        explicit_parents = tuple(dict.fromkeys(
            self.aliases[normalize_topic(value)]
            for value in values
            if normalize_topic(value) in self.aliases
        ))

        assignments: Dict[tuple[str, str], TopicAssignment] = {}
        for parent in explicit_parents:
            assignment = self._top_level(parent, source="controlled_alias")
            assignments[(assignment.level, assignment.key)] = assignment

        for value in values:
            normalized = normalize_topic(value)
            canonical = self.aliases.get(normalized)
            if canonical:
                continue

            hinted_parents = self._hinted_parents(value)
            parents = hinted_parents
            if not parents and len(explicit_parents) == 1:
                parents = explicit_parents
            elif not parents and len(explicit_parents) > 1:
                parents = explicit_parents

            if parents:
                for parent in parents:
                    parent_assignment = self._top_level(
                        parent,
                        source="controlled_parent_hint",
                        confidence=0.9,
                    )
                    assignments[(parent_assignment.level, parent_assignment.key)] = parent_assignment
                assignment = TopicAssignment(
                    name=value.strip(),
                    key=self._key(value),
                    status="active" if hinted_parents else "candidate",
                    source=source,
                    confidence=0.95 if hinted_parents else 0.6,
                    level="subtopic",
                    parent_names=parents,
                    parent_keys=tuple(self._key(parent) for parent in parents),
                )
            else:
                assignment = self._top_level(
                    value.strip() or "Unclassified",
                    source=source,
                    confidence=0.5 if normalized != "unclassified" else 1.0,
                )
            assignments[(assignment.level, assignment.key)] = assignment

        return list(assignments.values())

    def identity_for_entity(self, name: str) -> TopicAssignment:
        """Return the assignment matching an extracted topic entity itself."""
        assignments = self.assign(name, source="extraction")
        normalized = normalize_topic(name)
        return next(
            (
                assignment
                for assignment in assignments
                if normalize_topic(assignment.name) == normalized
            ),
            assignments[0],
        )


topic_taxonomy = TopicTaxonomy()
