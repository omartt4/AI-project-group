"""Extensible topic taxonomy with deterministic aliases and candidate topics."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List


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


@dataclass(frozen=True)
class TopicAssignment:
    name: str
    key: str
    status: str
    source: str
    confidence: float


class TopicTaxonomy:
    """Maps known aliases while preserving genuinely new topics as candidates."""

    def __init__(self) -> None:
        self.aliases = {
            normalize_topic(alias): canonical
            for canonical, aliases in SEED_TOPICS.items()
            for alias in aliases | {canonical}
        }

    def assign(self, raw_topics: str, source: str = "metadata") -> List[TopicAssignment]:
        values = [part.strip() for part in re.split(r"[;,|]", raw_topics or "") if part.strip()]
        if not values:
            values = ["Unclassified"]

        assignments: Dict[str, TopicAssignment] = {}
        for value in values:
            normalized = normalize_topic(value)
            canonical = self.aliases.get(normalized)
            if canonical:
                assignment = TopicAssignment(
                    name=canonical,
                    key=normalize_topic(canonical).replace(" ", "-"),
                    status="active",
                    source="controlled_alias",
                    confidence=1.0,
                )
            else:
                name = value.strip()
                assignment = TopicAssignment(
                    name=name,
                    key=normalized.replace(" ", "-") or "unclassified",
                    status="candidate" if normalized != "unclassified" else "active",
                    source=source,
                    confidence=0.5 if normalized != "unclassified" else 1.0,
                )
            assignments[assignment.key] = assignment
        return list(assignments.values())


topic_taxonomy = TopicTaxonomy()

