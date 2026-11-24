import csv
import json
import math
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = ROOT / "eval" / "questions_ar.jsonl"
RESULTS_PATH = ROOT / "eval" / "results_ar.json"
PER_QUESTION_PATH = ROOT / "eval" / "per_question_ar.csv"
SUMMARY_PATH = ROOT / "eval" / "summary_ar.csv"
CHROMA_PATH = ROOT / "chroma"
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "arabic_e5small_384_256_vector"
CONFIG_NAME = "arabic_e5small_384_256_vector"
TOP_K = 10


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def is_strict_hit(retrieved: dict, question: dict) -> bool:
    center = (question["gold_start"] + question["gold_end"]) // 2
    return (
        str(retrieved["book_id"]) == str(question["book_id"])
        and retrieved["char_start"] <= center <= retrieved["char_end"]
    )


def failure_class(
    top_result: dict | None,
    question: dict,
    ambiguous_question_ids: set[str],
) -> str:
    if top_result is not None and is_strict_hit(top_result, question):
        return "hit"
    if question["question_id"] in ambiguous_question_ids:
        return "ambiguous_question"
    if top_result is None or str(top_result["book_id"]) != str(question["book_id"]):
        return "wrong_book"
    return "right_book_wrong_chunk"


def first_strict_rank(results: list[dict], question: dict) -> int | None:
    for rank, result in enumerate(results, start=1):
        if is_strict_hit(result, question):
            return rank
    return None


def evaluate_retriever(
    retriever: str,
    questions: list[dict],
    ranked_results: list[list[dict]],
    ambiguous_question_ids: set[str],
) -> tuple[dict, list[dict]]:
    question_rows = []
    ranks = []
    failure_counts = {
        "hit": 0,
        "ambiguous_question": 0,
        "right_book_wrong_chunk": 0,
        "wrong_book": 0,
    }

    for question, results in zip(questions, ranked_results):
        rank = first_strict_rank(results, question)
        ranks.append(rank)
        category = failure_class(
            results[0] if results else None,
            question,
            ambiguous_question_ids,
        )
        failure_counts[category] += 1
        top_result = results[0] if results else {}
        row = {
            "config_name": CONFIG_NAME,
            "retriever": retriever,
            "question_id": question["question_id"],
            "book_id": question["book_id"],
            "question": question["question"],
            "gold_start": question["gold_start"],
            "gold_end": question["gold_end"],
            "strict_rank": rank if rank is not None else "",
            "hit@1": int(rank == 1),
            "hit@5": int(rank is not None and rank <= 5),
            "hit@10": int(rank is not None and rank <= 10),
            "reciprocal_rank": 1 / rank if rank is not None else 0.0,
            "top1_book_id": top_result.get("book_id", ""),
            "top1_chunk_id": top_result.get("chunk_id", ""),
            "failure_class_hit1": category,
            "ambiguous_by_cross_book_answer_quote": int(
                question["question_id"] in ambiguous_question_ids
            ),
        }
        question_rows.append(row)

    total = len(questions)
    hits = {
        k: sum(rank is not None and rank <= k for rank in ranks)
        for k in (1, 5, 10)
    }
    ci_1 = wilson_interval(hits[1], total)
    ci_10 = wilson_interval(hits[10], total)
    metrics = {
        "config_name": CONFIG_NAME,
        "retriever": retriever,
        "n": total,
        "hit@1": hits[1] / total,
        "hit@1_count": hits[1],
        "hit@1_ci95": list(ci_1),
        "hit@5": hits[5] / total,
        "hit@5_count": hits[5],
        "hit@10": hits[10] / total,
        "hit@10_count": hits[10],
        "hit@10_ci95": list(ci_10),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total,
        "failure_counts_at_1": failure_counts,
    }
    return metrics, question_rows


def main() -> None:
    output_paths = (RESULTS_PATH, PER_QUESTION_PATH, SUMMARY_PATH)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evaluation files: {existing}")

    chunks = load_jsonl(CHUNKS_PATH)
    questions = load_jsonl(QUESTIONS_PATH)
    if not questions:
        raise ValueError("No Arabic evaluation questions were found.")
    if len(questions) < 15:
        raise ValueError(
            f"Only {len(questions)} questions found; expected at least 15."
        )
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for question in questions:
        chunk = chunk_by_id.get(question["chunk_id"])
        if chunk is None or str(chunk["book_id"]) != str(question["book_id"]):
            raise ValueError(f"Invalid gold chunk reference: {question['question_id']}")

    cleaned_by_book = {
        book_id: (ROOT / "cleaned" / f"book_{book_id}.txt").read_text(
            encoding="utf-8"
        )
        for book_id in ("140", "1144")
    }
    ambiguous_question_ids = set()
    for question in questions:
        other_book_id = "1144" if str(question["book_id"]) == "140" else "140"
        other_text = cleaned_by_book[other_book_id]
        if question["answer_quote"] in other_text:
            ambiguous_question_ids.add(question["question_id"])

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    if collection.count() != len(chunks):
        raise ValueError(
            f"Chroma has {collection.count()} entries; expected {len(chunks)}."
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if model.get_embedding_dimension() != 384:
        raise ValueError("The Arabic evaluation model must have 384 dimensions.")
    query_texts = [f"query: {question['question']}" for question in questions]
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vector_results = []
    for query_embedding in query_embeddings:
        retrieved = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(TOP_K, len(chunks)),
            include=["metadatas"],
        )
        ranked = []
        for chunk_id, metadata in zip(
            retrieved["ids"][0], retrieved["metadatas"][0]
        ):
            ranked.append(
                {
                    "chunk_id": chunk_id,
                    "book_id": str(metadata["book_id"]),
                    "char_start": int(metadata["char_start"]),
                    "char_end": int(metadata["char_end"]),
                }
            )
        vector_results.append(ranked)

    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    bm25_results = []
    for question in questions:
        scores = bm25.get_scores(tokenize(question["question"]))
        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-scores[index], chunks[index]["chunk_id"]),
        )[:TOP_K]
        bm25_results.append(
            [
                {
                    "chunk_id": chunks[index]["chunk_id"],
                    "book_id": str(chunks[index]["book_id"]),
                    "char_start": chunks[index]["char_start"],
                    "char_end": chunks[index]["char_end"],
                }
                for index in ranked_indices
            ]
        )

    vector_metrics, vector_rows = evaluate_retriever(
        "vector_e5",
        questions,
        vector_results,
        ambiguous_question_ids,
    )
    bm25_metrics, bm25_rows = evaluate_retriever(
        "bm25",
        questions,
        bm25_results,
        ambiguous_question_ids,
    )
    all_rows = vector_rows + bm25_rows
    result_data = {
        "config_name": CONFIG_NAME,
        "model": MODEL_NAME,
        "dimension": 384,
        "chunk_size_tokens": 256,
        "query_prefix": "query: ",
        "strict_hit_rule": (
            "retrieved chunk has same book_id and contains the gold span center"
        ),
        "n_questions": len(questions),
        "small_sample_warning": (
            "With this sample size, differences of 1-2 questions are not meaningful."
        ),
        "ambiguous_question_heuristic": (
            "The exact gold answer_quote also appears in the other book's cleaned text."
        ),
        "retrievers": {
            "vector_e5": vector_metrics,
            "bm25": bm25_metrics,
        },
        "local_embedding_cost_usd": 0,
    }

    with RESULTS_PATH.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result_data, output, ensure_ascii=False, indent=2)
        output.write("\n")

    per_question_fields = list(all_rows[0].keys())
    with PER_QUESTION_PATH.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=per_question_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for metrics in (vector_metrics, bm25_metrics):
        failures = metrics["failure_counts_at_1"]
        summary_rows.append(
            {
                "config_name": CONFIG_NAME,
                "retriever": metrics["retriever"],
                "n": metrics["n"],
                "hit@1_count": metrics["hit@1_count"],
                "hit@1": metrics["hit@1"],
                "hit@1_ci95_lower": metrics["hit@1_ci95"][0],
                "hit@1_ci95_upper": metrics["hit@1_ci95"][1],
                "hit@5_count": metrics["hit@5_count"],
                "hit@5": metrics["hit@5"],
                "hit@10_count": metrics["hit@10_count"],
                "hit@10": metrics["hit@10"],
                "hit@10_ci95_lower": metrics["hit@10_ci95"][0],
                "hit@10_ci95_upper": metrics["hit@10_ci95"][1],
                "mrr": metrics["mrr"],
                "ambiguous_question": failures["ambiguous_question"],
                "right_book_wrong_chunk": failures["right_book_wrong_chunk"],
                "wrong_book": failures["wrong_book"],
            }
        )
    with SUMMARY_PATH.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for metrics in (vector_metrics, bm25_metrics):
        print(
            f"{metrics['retriever']}: n={metrics['n']} "
            f"Hit@1={metrics['hit@1_count']}/{metrics['n']} "
            f"Hit@5={metrics['hit@5_count']}/{metrics['n']} "
            f"Hit@10={metrics['hit@10_count']}/{metrics['n']} "
            f"MRR={metrics['mrr']:.3f}"
        )
    print(f"questions={QUESTIONS_PATH}")
    print(f"results={RESULTS_PATH}")
    print(f"per_question={PER_QUESTION_PATH}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
