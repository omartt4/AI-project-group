"""
Benchmark script comparing sequential versus concurrent execution for retrieval and ingestion steps.
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Ensure project src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.config import settings
from graph_rag.retriever.query_classifier import query_classifier


def benchmark_tasks(n_tasks: int = 40):
    print(f"--- Benchmarking Sequential vs Concurrent Execution ({n_tasks} tasks) ---")
    queries = [
        "What is Transformer?",
        "Compare LSTM vs BERT",
        "Summarize speaker diarization research",
        "What datasets were used in relation extraction?",
    ] * (n_tasks // 4 + 1)
    sample_queries = queries[:n_tasks]

    # 1. Sequential execution
    t0 = time.perf_counter()
    seq_results = []
    for q in sample_queries:
        seq_results.append(query_classifier.classify(q))
    seq_elapsed = time.perf_counter() - t0
    print(f"Sequential Execution Time: {seq_elapsed:.4f}s ({n_tasks / seq_elapsed:.1f} tasks/sec)")

    # 2. Concurrent execution with configured max_concurrency
    max_workers = settings.LANGCHAIN_MAX_CONCURRENCY
    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        conc_results = list(executor.map(query_classifier.classify, sample_queries))
    conc_elapsed = time.perf_counter() - t1
    print(f"Concurrent Execution Time (max_concurrency={max_workers}): {conc_elapsed:.4f}s ({n_tasks / conc_elapsed:.1f} tasks/sec)")

    speedup = seq_elapsed / conc_elapsed if conc_elapsed > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    assert len(seq_results) == len(conc_results) == n_tasks
    print("✅ Benchmark completed successfully!")


if __name__ == "__main__":
    benchmark_tasks()
