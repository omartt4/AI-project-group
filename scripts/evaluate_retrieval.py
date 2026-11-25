"""Run deterministic offline evaluation for vector, graph, or hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.evaluation.retrieval_metrics import (  # noqa: E402
    evaluate_question,
    load_gold_jsonl,
    summarize_results,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    return {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_")
    }


def _empty_payload() -> dict[str, list[dict[str, Any]]]:
    return {"chunks": [], "concepts": [], "graph_facts": []}


def _retrieve_for_modes(question: str, modes: list[str]) -> dict[str, dict[str, Any]]:
    # Import lazily so ``--help`` and metric unit tests need no DB/model startup.
    from graph_rag.retriever.hybrid_retriever import hybrid_retriever

    classification = hybrid_retriever.classifier.classify(question)
    payloads = {mode: _empty_payload() for mode in modes}

    if "hybrid" in modes or len(modes) > 1:
        context = hybrid_retriever.adaptive_retrieve(
            question,
            classification=classification,
        )
        shared = {
            "chunks": list(context.chunks),
            "concepts": [_as_dict(item) for item in context.discovered_concepts],
            "graph_facts": list(context.graph_facts),
        }
        if "hybrid" in modes:
            payloads["hybrid"] = shared
        if "vector" in modes:
            payloads["vector"] = {**_empty_payload(), "chunks": shared["chunks"]}
        if "graph" in modes:
            payloads["graph"] = {
                **_empty_payload(),
                "concepts": shared["concepts"],
                "graph_facts": shared["graph_facts"],
            }
        return payloads

    mode = modes[0]
    if mode == "vector":
        payloads[mode]["chunks"] = hybrid_retriever._retrieve_vectors(
            question,
            classification,
        )
    elif mode == "graph":
        discovery = hybrid_retriever._discover_graph(question, classification)
        payloads[mode]["concepts"] = [_as_dict(item) for item in discovery.concepts]
        payloads[mode]["graph_facts"] = list(discovery.relationships)
    else:
        context = hybrid_retriever.adaptive_retrieve(
            question,
            classification=classification,
        )
        payloads[mode] = {
            "chunks": list(context.chunks),
            "concepts": [_as_dict(item) for item in context.discovered_concepts],
            "graph_facts": list(context.graph_facts),
        }
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval against a gold JSONL dataset without calling the "
            "chat or evaluation model. Vector/hybrid modes still embed each question."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation_datasets" / "afaq57_retrieval_gold.jsonl",
        help="Gold JSONL dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=("vector", "graph", "hybrid", "all"),
        default="all",
        help="Retrieval path to evaluate. 'all' runs hybrid once and scores its two branches.",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="Threshold used only to identify false retrieval on unanswerable questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation_reports" / "retrieval_report.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions for a cheap smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_rows = load_gold_jsonl(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        gold_rows = gold_rows[: args.limit]

    modes = ["vector", "graph", "hybrid"] if args.mode == "all" else [args.mode]
    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

    for index, gold in enumerate(gold_rows, start=1):
        print(f"[{index}/{len(gold_rows)}] {gold['question_id']}: {gold['question']}")
        try:
            payloads = _retrieve_for_modes(str(gold["question"]), modes)
            for mode in modes:
                payload = payloads[mode]
                row = evaluate_question(
                    gold,
                    chunks=payload["chunks"],
                    concepts=payload["concepts"],
                    graph_facts=payload["graph_facts"],
                    ks=args.k,
                    min_similarity=args.min_similarity,
                )
                # Page-ranking metrics are defined for vector results; graph metrics
                # are defined for graph results. Hybrid evaluation supports both.
                if mode == "graph":
                    row["mrr"] = None
                    for k in args.k:
                        row[f"hit_rate@{k}"] = None
                        row[f"precision@{k}"] = None
                        row[f"recall@{k}"] = None
                elif mode == "vector":
                    row["entity_recall"] = None
                    row["relation_recall"] = None
                row["mode"] = mode
                results[mode].append(row)
        except Exception as exc:
            for mode in modes:
                row = evaluate_question(gold, ks=args.k)
                row["mode"] = mode
                row["error"] = f"{type(exc).__name__}: {exc}"
                results[mode].append(row)
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "configuration": {
            "mode": args.mode,
            "k": sorted(set(args.k)),
            "min_similarity": args.min_similarity,
            "chat_model_called": False,
            "evaluation_model_called": False,
            "embedding_note": "Vector and hybrid retrieval embed each question; graph-only does not.",
        },
        "modes": {
            mode: {
                "summary": summarize_results(rows, args.k),
                "questions": rows,
            }
            for mode, rows in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRetrieval evaluation summary")
    for mode in modes:
        summary = report["modes"][mode]["summary"]
        print(
            f"  {mode}: Hit@5={summary.get('hit_rate@5')} "
            f"Recall@5={summary.get('recall@5')} "
            f"Precision@5={summary.get('precision@5')} "
            f"MRR={summary.get('mrr')} "
            f"EntityRecall={summary.get('entity_recall')} "
            f"RelationRecall={summary.get('relation_recall')} "
            f"FalseRetrievalRate={summary.get('false_retrieval_rate')}"
        )
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
