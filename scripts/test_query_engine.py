"""
Test script for verifying QueryEngine end-to-end with LLM synthesis.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.retriever.query_engine import query_engine


def main():
    print("Testing QueryEngine end-to-end...")
    question = "What is BERT and how is attention used in it?"
    # Test without eval to be fast
    resp = query_engine.query(question=question, run_evaluation=False)

    print(f"\nQuestion: {resp.question}")
    print(f"Classification: {resp.classification.query_class.value if resp.classification else 'None'}")
    print(f"Discovered Concepts Count: {resp.discovered_concepts_count}")
    print(f"Unique Papers: {resp.unique_papers_count}")
    print(f"Sources: {resp.sources}")
    print(f"Latency: Retrieval={resp.latency.retrieval_ms}ms, Gen={resp.latency.generation_ms}ms, Total={resp.latency.total_ms}ms")
    print(f"\nAnswer Preview:\n{resp.answer[:400]}...")
    assert len(resp.answer) > 20
    print("\nQueryEngine test passed successfully!")


if __name__ == "__main__":
    main()

