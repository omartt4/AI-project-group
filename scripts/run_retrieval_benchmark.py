"""
Retrieval Benchmark Script.

Runs end-to-end tests on the Graph RAG system with various query types
to measure retrieval quality, latency, and system health.

Usage:
    uv run python scripts/run_retrieval_benchmark.py
"""

import time
import logging
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, field
from tabulate import tabulate

# Set environment variable to disable PyMuPDF layout analyzer (reduces memory usage)
import os
os.environ["PYMUPDF_DISABLE_LAYOUT_ANALYZER"] = "1"

try:
    from graph_rag.retriever.query_engine import query_engine
    from graph_rag.retriever.query_classifier import QueryClass
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nThis benchmark requires the Graph RAG system to be properly set up.")
    print("If you're experiencing disk space issues, please:")
    print("1. Free up space on your C: drive")
    print("2. Or set environment variables to use D: drive for data storage")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark query."""
    query: str
    query_type: str
    success: bool
    latency_ms: int
    chunks_retrieved: int
    unique_papers: int
    concepts_discovered: int
    graph_facts: int
    evidence_quality: float
    insufficient_evidence: bool
    detected_language: str
    error_message: str = ""


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    avg_chunks: float
    avg_concepts: float
    avg_evidence_quality: float
    insufficient_evidence_count: int


class RetrievalBenchmark:
    """End-to-end retrieval benchmark suite."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_test_queries(self) -> None:
        """Run a suite of test queries covering different query types."""
        
        test_queries = [
            # Factual queries
            ("What is attention mechanism in transformers?", "factual"),
            ("Define cross-entropy loss", "factual"),
            
            # Comparative queries
            ("Compare BERT and GPT architectures", "comparative"),
            ("Difference between CNN and RNN", "comparative"),
            
            # Relational queries
            ("Who proposed the Transformer architecture?", "relational"),
            ("What datasets were used to train BERT?", "relational"),
            
            # Exploratory queries
            ("Explain recent advances in computer vision", "exploratory"),
            ("What are the main challenges in NLP?", "exploratory"),
            
            # Arabic queries (multilingual test)
            ("ما هو التعلم العميق؟", "arabic"),
            ("قارن بين الشبكات العصبية والشبكات التلافيفية", "arabic_comparative"),
        ]

        logger.info(f"Running {len(test_queries)} benchmark queries...")
        logger.info("=" * 60)

        for query, query_type in test_queries:
            try:
                start_time = time.time()
                response = query_engine.query(
                    question=query,
                    run_evaluation=False,  # Skip LLM evaluation for speed
                )
                elapsed_ms = int((time.time() - start_time) * 1000)

                result = BenchmarkResult(
                    query=query,
                    query_type=query_type,
                    success=True,
                    latency_ms=elapsed_ms,
                    chunks_retrieved=len(response.chunks),
                    unique_papers=response.unique_papers_count,
                    concepts_discovered=response.discovered_concepts_count,
                    graph_facts=len(response.graph_facts),
                    evidence_quality=response.evidence_quality_score,
                    insufficient_evidence=response.insufficient_evidence,
                    detected_language=response.detected_language,
                )
                
                logger.info(f"✓ {query_type}: {query[:50]}... ({elapsed_ms}ms)")
                
            except Exception as e:
                result = BenchmarkResult(
                    query=query,
                    query_type=query_type,
                    success=False,
                    latency_ms=0,
                    chunks_retrieved=0,
                    unique_papers=0,
                    concepts_discovered=0,
                    graph_facts=0,
                    evidence_quality=0.0,
                    insufficient_evidence=True,
                    detected_language="unknown",
                    error_message=str(e),
                )
                logger.error(f"✗ {query_type}: {query[:50]}... - {e}")

            self.results.append(result)
            time.sleep(0.5)  # Brief pause between queries

    def generate_summary(self) -> BenchmarkSummary:
        """Generate summary statistics from benchmark results."""
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return BenchmarkSummary(
                total_queries=len(self.results),
                successful_queries=0,
                failed_queries=len(self.results),
                avg_latency_ms=0.0,
                avg_chunks=0.0,
                avg_concepts=0.0,
                avg_evidence_quality=0.0,
                insufficient_evidence_count=0,
            )

        return BenchmarkSummary(
            total_queries=len(self.results),
            successful_queries=len(successful),
            failed_queries=len(self.results) - len(successful),
            avg_latency_ms=sum(r.latency_ms for r in successful) / len(successful),
            avg_chunks=sum(r.chunks_retrieved for r in successful) / len(successful),
            avg_concepts=sum(r.concepts_discovered for r in successful) / len(successful),
            avg_evidence_quality=sum(r.evidence_quality for r in successful) / len(successful),
            insufficient_evidence_count=sum(1 for r in successful if r.insufficient_evidence),
        )

    def print_results(self) -> None:
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 100)
        print("RETRIEVAL BENCHMARK RESULTS")
        print("=" * 100)
        
        # Detailed results table
        table_data = []
        for r in self.results:
            status = "✓" if r.success else "✗"
            table_data.append([
                status,
                r.query_type,
                r.query[:40] + "..." if len(r.query) > 40 else r.query,
                r.detected_language,
                r.latency_ms,
                r.chunks_retrieved,
                r.unique_papers,
                r.concepts_discovered,
                f"{r.evidence_quality:.2f}",
                "⚠️" if r.insufficient_evidence else "",
            ])

        headers = [
            "Status",
            "Type",
            "Query",
            "Lang",
            "Latency(ms)",
            "Chunks",
            "Papers",
            "Concepts",
            "Quality",
            "Low Evid",
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Summary statistics
        summary = self.generate_summary()
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        print(f"Total Queries:        {summary.total_queries}")
        print(f"Successful:           {summary.successful_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)")
        print(f"Failed:               {summary.failed_queries} ({summary.failed_queries/summary.total_queries*100:.1f}%)")
        print(f"Avg Latency:          {summary.avg_latency_ms:.0f}ms")
        print(f"Avg Chunks Retrieved: {summary.avg_chunks:.1f}")
        print(f"Avg Concepts Found:   {summary.avg_concepts:.1f}")
        print(f"Avg Evidence Quality: {summary.avg_evidence_quality:.2f}")
        print(f"Low Evidence Count:   {summary.insufficient_evidence_count}")
        print("=" * 100)

    def print_failures(self) -> None:
        """Print detailed information about failed queries."""
        failures = [r for r in self.results if not r.success]
        
        if failures:
            print("\n" + "=" * 100)
            print("FAILED QUERIES")
            print("=" * 100)
            for f in failures:
                print(f"\nQuery: {f.query}")
                print(f"Type:  {f.query_type}")
                print(f"Error: {f.error_message}")
            print("=" * 100)


def main():
    """Run the retrieval benchmark."""
    print("\n🚀 Starting Graph RAG Retrieval Benchmark...")
    print("This will test various query types and measure system performance.\n")
    
    benchmark = RetrievalBenchmark()
    
    try:
        benchmark.run_test_queries()
        benchmark.print_results()
        benchmark.print_failures()
        
        summary = benchmark.generate_summary()
        
        if summary.failed_queries > 0:
            logger.warning(f"Benchmark completed with {summary.failed_queries} failures")
            return 1
        else:
            logger.info("Benchmark completed successfully")
            return 0
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
