"""Open-schema topic normalization and evidence-gated hierarchy assignment."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

_GENERIC_TOPIC_NAMES = {
    "abstract",
    "acknowledgments",
    "background",
    "conclusion",
    "discussion",
    "introduction",
    "method",
    "methods",
    "references",
    "results",
}


def normalize_topic(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()


@dataclass(frozen=True)
class TopicAssignment:
    """A topic with an automatically selected, evidence-backed placement."""

    name: str
    key: str
    status: str
    source: str
    confidence: float
    level: str = "top_level"
    parent_names: Tuple[str, ...] = ()
    parent_keys: Tuple[str, ...] = ()
    evidence: str = ""
    aliases: Tuple[str, ...] = ()


class TopicTaxonomy:
    """Normalize extracted topics without a domain-specific allowlist."""

    def __init__(
        self,
        *,
        merge_threshold: float = 0.94,
        minimum_confidence: float = 0.8,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.minimum_confidence = minimum_confidence

    @staticmethod
    def _key(name: str) -> str:
        return normalize_topic(name).replace(" ", "-") or "topic"

    @staticmethod
    def _existing_labels(topic: dict) -> List[str]:
        aliases = topic.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return [
            str(label)
            for label in [topic.get("name") or "", *aliases]
            if label
        ]

    def _resolve_existing(
        self,
        name: str,
        existing_topics: List[dict],
    ) -> tuple[str, bool, bool]:
        normalized = normalize_topic(name)
        exact = [
            str(topic["name"])
            for topic in existing_topics
            if normalized
            in {
                normalize_topic(label)
                for label in self._existing_labels(topic)
            }
        ]
        exact = list(dict.fromkeys(exact))
        if len(exact) == 1:
            return exact[0], True, False
        if len(exact) > 1:
            return name, False, True

        candidates = sorted(
            [
                (
                    max(
                        SequenceMatcher(
                            None,
                            normalized,
                            normalize_topic(label),
                        ).ratio()
                        for label in self._existing_labels(topic)
                    ),
                    str(topic["name"]),
                )
                for topic in existing_topics
                if topic.get("name")
            ],
            reverse=True,
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0] >= self.merge_threshold
        ]
        if not candidates:
            return name, False, False
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
            return name, False, True
        return candidates[0][1], True, False

    def assign(
        self,
        raw_topics: str = "",
        source: str = "model_extraction",
        proposals: List[dict] | None = None,
        existing_topics: List[dict] | None = None,
        entity_names: Iterable[str] = (),
        diagnostics: List[dict] | None = None,
    ) -> List[TopicAssignment]:
        """Return only grounded, high-confidence topics; uncertain cases stay diagnostic."""
        del raw_topics  # Metadata-derived topic strings are hints, not graph evidence.
        proposals = proposals or []
        existing_topics = existing_topics or []
        entity_keys = {normalize_topic(name) for name in entity_names if name}
        topic_names: Dict[str, tuple[str, dict, float]] = {}

        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            name = " ".join(str(proposal.get("name") or "").split())
            evidence = " ".join(str(proposal.get("evidence") or "").split())
            normalized_name = normalize_topic(name)
            if (
                len(name) < 2
                or len(name) > 120
                or len(name.split()) > 12
                or normalized_name in _GENERIC_TOPIC_NAMES
                or not evidence
            ):
                continue
            try:
                confidence = min(1.0, max(0.0, float(proposal.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            canonical, matched, ambiguous = self._resolve_existing(name, existing_topics)
            if ambiguous:
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "diagnostic_type": "ambiguous_topic_resolution",
                            "name": name,
                            "evidence": evidence,
                            "confidence": confidence,
                            "candidates": [
                                str(topic.get("name") or "")
                                for topic in existing_topics
                                if topic.get("name")
                            ],
                        }
                    )
                continue
            normalized = normalize_topic(canonical)
            if not normalized or normalized in entity_keys:
                continue
            for alias in proposal.get("aliases", []) or []:
                if isinstance(alias, str) and normalize_topic(alias):
                    topic_names[normalize_topic(alias)] = (
                        canonical,
                        proposal,
                        confidence,
                    )
            topic_names[normalized] = (canonical, proposal, confidence)

        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            evidence = " ".join(str(proposal.get("evidence") or "").split())
            try:
                confidence = max(
                    0.0,
                    min(1.0, float(proposal.get("confidence", 0.0))),
                )
            except (TypeError, ValueError):
                confidence = 0.0
            parents = proposal.get("parent_names") or []
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                parent_name = " ".join(str(parent).split())
                normalized_parent = normalize_topic(parent_name)
                if (
                    len(parent_name) < 2
                    or len(parent_name) > 120
                    or len(parent_name.split()) > 12
                    or normalized_parent in _GENERIC_TOPIC_NAMES
                    or normalized_parent in entity_keys
                    or normalized_parent in topic_names
                ):
                    continue
                topic_names[normalized_parent] = (
                    parent_name,
                    {
                        "name": parent_name,
                        "evidence": evidence,
                        "confidence": confidence,
                        "parent_names": [],
                    },
                    confidence,
                )

        assignments: Dict[tuple[str, str], TopicAssignment] = {}
        for canonical, proposal, confidence in topic_names.values():
            evidence = " ".join(str(proposal.get("evidence") or "").split())
            key = self._key(canonical)
            if confidence < self.minimum_confidence:
                continue
            parent_names: List[str] = []
            raw_parents = proposal.get("parent_names") or []
            if isinstance(raw_parents, str):
                raw_parents = [raw_parents]
            for parent_name in raw_parents:
                if not isinstance(parent_name, str):
                    continue
                normalized_parent = normalize_topic(parent_name)
                if (
                    normalized_parent
                    and normalized_parent != normalize_topic(canonical)
                    and normalized_parent in topic_names
                ):
                    parent_names.append(topic_names[normalized_parent][0])
            parent_names = list(dict.fromkeys(parent_names))
            level = "subtopic" if parent_names else "top_level"
            raw_aliases = proposal.get("aliases") or []
            if not isinstance(raw_aliases, list):
                raw_aliases = [raw_aliases]
            if matched and normalize_topic(name) != normalize_topic(canonical):
                raw_aliases.append(name)
            aliases = tuple(
                dict.fromkeys(
                    " ".join(str(alias).split())
                    for alias in raw_aliases
                    if isinstance(alias, str)
                    and normalize_topic(alias)
                    and normalize_topic(alias) != normalize_topic(canonical)
                )
            )
            assignment = TopicAssignment(
                name=canonical,
                key=key,
                status="active",
                source=source,
                confidence=confidence,
                level=level,
                parent_names=tuple(parent_names),
                parent_keys=tuple(self._key(parent) for parent in parent_names),
                evidence=evidence,
                aliases=aliases,
            )
            assignment_key = (level, key)
            current = assignments.get(assignment_key)
            if current:
                parent_names = tuple(
                    dict.fromkeys(current.parent_names + assignment.parent_names)
                )
                assignment = replace(
                    assignment,
                    confidence=max(current.confidence, assignment.confidence),
                    parent_names=parent_names,
                    parent_keys=tuple(self._key(parent) for parent in parent_names),
                    evidence=(
                        assignment.evidence
                        if assignment.confidence > current.confidence
                        else current.evidence
                    ),
                    aliases=tuple(
                        dict.fromkeys(current.aliases + assignment.aliases)
                    ),
                )
            assignments[assignment_key] = assignment

        # Ensure each accepted parent is represented as a top-level topic.
        for assignment in list(assignments.values()):
            for parent in assignment.parent_names:
                parent_key = self._key(parent)
                assignments.setdefault(
                    ("top_level", parent_key),
                    TopicAssignment(
                        name=parent,
                        key=parent_key,
                        status="active",
                        source=source,
                        confidence=assignment.confidence,
                        evidence=assignment.evidence,
                    ),
                )
        return list(assignments.values())

    def identity_for_entity(self, name: str) -> TopicAssignment:
        """Create a stable identity for a topic node without restricting its label."""
        normalized = normalize_topic(name)
        return TopicAssignment(
            name=" ".join(name.split()),
            key=self._key(name),
            status="active",
            source="extraction",
            confidence=1.0,
            evidence="",
        )


topic_taxonomy = TopicTaxonomy()
