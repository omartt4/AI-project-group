"""Deterministic offline retrieval metrics backed by a human-reviewed gold set."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def normalize_label(value: Any) -> str:
    """Normalize English/Arabic labels for tolerant deterministic comparison."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace("ـ", "")
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def labels_match(expected: Any, actual: Any) -> bool:
    """Match normalized labels, allowing a meaningful label to contain the other."""
    left = normalize_label(expected)
    right = normalize_label(actual)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and shorter in longer


def parse_page_spec(value: Any) -> list[int]:
    """Parse page specifications such as ``1,4-6,74`` into unique page numbers."""
    if value in (None, ""):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        pages: list[int] = []
        for item in value:
            pages.extend(parse_page_spec(item))
        return sorted(set(pages))

    pages = []
    for part in re.split(r"[,;]", str(value)):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            if end < start:
                start, end = end, start
            pages.extend(range(start, end + 1))
        elif part.isdigit():
            pages.append(int(part))
    return sorted(set(pages))


def load_gold_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an offline evaluation dataset."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            for required in ("question_id", "question", "answerable"):
                if required not in row:
                    raise ValueError(f"Line {line_number} is missing {required!r}")
            rows.append(row)
    if not rows:
        raise ValueError("Gold dataset is empty")
    return rows


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _ranked_unique_pages(chunks: Sequence[dict[str, Any]]) -> list[tuple[str, int]]:
    ranked: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        metadata = _metadata(chunk)
        document = normalize_label(
            metadata.get("document_name")
            or metadata.get("filename")
            or metadata.get("source")
        )
        raw_page = metadata.get("page", metadata.get("page_number"))
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            continue
        item = (document, page)
        if document and item not in seen:
            seen.add(item)
            ranked.append(item)
    return ranked


def _expected_pages(gold: dict[str, Any]) -> set[tuple[str, int]]:
    document = normalize_label(gold.get("expected_document"))
    pages = parse_page_spec(gold.get("relevant_pdf_pages", gold.get("relevant_pages")))
    return {(document, page) for page in pages if document}


def _page_matches(expected: tuple[str, int], actual: tuple[str, int]) -> bool:
    return expected[1] == actual[1] and labels_match(expected[0], actual[0])


def _first_relevant_rank(
    ranked: Sequence[tuple[str, int]],
    expected: set[tuple[str, int]],
) -> int | None:
    for rank, actual in enumerate(ranked, start=1):
        if any(_page_matches(item, actual) for item in expected):
            return rank
    return None


def _matched_gold_pages(
    ranked: Sequence[tuple[str, int]],
    expected: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    return {
        gold_item
        for gold_item in expected
        if any(_page_matches(gold_item, actual) for actual in ranked)
    }


def _concept_names(
    concepts: Sequence[dict[str, Any]],
    graph_facts: Sequence[dict[str, Any]],
) -> list[str]:
    names: list[str] = []
    for concept in concepts:
        name = concept.get("name")
        if name:
            names.append(str(name))
    for fact in graph_facts:
        for key in ("source", "target", "source_name", "target_name"):
            if fact.get(key):
                names.append(str(fact[key]))
    return names


def _relation_names(graph_facts: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for fact in graph_facts:
        for key in ("relation", "relation_type", "type"):
            if fact.get(key):
                names.append(str(fact[key]))
                break
    return names


def _recall(expected: Iterable[Any], actual: Iterable[Any]) -> float | None:
    expected_values = [value for value in expected if normalize_label(value)]
    if not expected_values:
        return None
    actual_values = [value for value in actual if normalize_label(value)]
    matched = sum(
        1
        for expected_value in expected_values
        if any(labels_match(expected_value, actual_value) for actual_value in actual_values)
    )
    return matched / len(expected_values)


def _as_values(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def evaluate_question(
    gold: dict[str, Any],
    *,
    chunks: Sequence[dict[str, Any]] = (),
    concepts: Sequence[dict[str, Any]] = (),
    graph_facts: Sequence[dict[str, Any]] = (),
    ks: Sequence[int] = (3, 5, 10),
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Compare one retrieval result with its gold evidence deterministically."""
    answerable = bool(gold.get("answerable"))
    ranked_pages = _ranked_unique_pages(chunks)
    expected_pages = _expected_pages(gold)
    result: dict[str, Any] = {
        "question_id": gold["question_id"],
        "question": gold["question"],
        "answerable": answerable,
        "retrieved_chunk_count": len(chunks),
        "retrieved_unique_pages": [
            {"document": document, "page": page}
            for document, page in ranked_pages
        ],
        "retrieved_concepts": _concept_names(concepts, graph_facts),
        "retrieved_relations": _relation_names(graph_facts),
    }

    if answerable:
        first_rank = _first_relevant_rank(ranked_pages, expected_pages)
        result["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
        for k in sorted(set(int(value) for value in ks if int(value) > 0)):
            top_k = ranked_pages[:k]
            matched = _matched_gold_pages(top_k, expected_pages)
            result[f"hit_rate@{k}"] = 1.0 if matched else 0.0
            result[f"precision@{k}"] = len(matched) / k
            result[f"recall@{k}"] = (
                len(matched) / len(expected_pages) if expected_pages else None
            )
        result["entity_recall"] = _recall(
            _as_values(gold.get("expected_entities")),
            _concept_names(concepts, graph_facts),
        )
        result["relation_recall"] = _recall(
            _as_values(
                gold.get("expected_relations")
                or gold.get("expected_relation_or_theme")
            ),
            _relation_names(graph_facts),
        )
        result["false_retrieval"] = None
    else:
        strong_vector_hit = any(
            float(chunk.get("similarity_score") or 0.0) >= min_similarity
            for chunk in chunks
        )
        graph_hit = bool(concepts or graph_facts)
        result.update(
            {
                "mrr": None,
                "entity_recall": None,
                "relation_recall": None,
                "false_retrieval": bool(strong_vector_hit or graph_hit),
            }
        )
        for k in sorted(set(int(value) for value in ks if int(value) > 0)):
            result[f"hit_rate@{k}"] = None
            result[f"precision@{k}"] = None
            result[f"recall@{k}"] = None
    return result


def summarize_results(
    rows: Sequence[dict[str, Any]],
    ks: Sequence[int] = (3, 5, 10),
) -> dict[str, Any]:
    """Create macro averages without silently replacing unavailable metrics by zero."""
    metric_names = ["mrr", "entity_recall", "relation_recall"]
    for k in sorted(set(int(value) for value in ks if int(value) > 0)):
        metric_names.extend((f"hit_rate@{k}", f"precision@{k}", f"recall@{k}"))

    summary: dict[str, Any] = {
        "questions": len(rows),
        "answerable_questions": sum(bool(row.get("answerable")) for row in rows),
        "unanswerable_questions": sum(not bool(row.get("answerable")) for row in rows),
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None
        summary[f"{metric}_evaluated_questions"] = len(values)

    false_values = [
        bool(row["false_retrieval"])
        for row in rows
        if row.get("false_retrieval") is not None
    ]
    summary["false_retrieval_rate"] = (
        sum(false_values) / len(false_values) if false_values else None
    )
    summary["false_retrieval_evaluated_questions"] = len(false_values)
    summary["errors"] = sum(bool(row.get("error")) for row in rows)
    return summary
