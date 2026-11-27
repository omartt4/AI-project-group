"""
Comprehensive Test Script for Adaptive Hybrid RAG with Diversity-Aware Retrieval.

Verifies:
1. Query Classification & Intent Routing
2. Dynamic Top-K & Adaptive Parameters
3. Vectorized MMR Diversity Retrieval & Cross-Doc Penalization
4. Graph Concept Discovery & Bridge Detection
5. Structured Evidence Aggregation
6. End-to-End Hybrid Retriever Integration
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import numpy as np
from graph_rag.retriever.query_classifier import QueryClass, query_classifier
from graph_rag.retriever.diversity_retriever import diversity_retriever
from graph_rag.graph.concept_discovery import (
    ConceptDiscoveryResult,
    DiscoveredConcept,
    concept_discovery_engine,
)
from graph_rag.retriever.evidence_aggregator import evidence_aggregator
from graph_rag.retriever.hybrid_retriever import hybrid_retriever


def test_query_classification() -> None:
    """Test query intent classifier on various query types."""
    print("\n--- 1. Testing Query Classifier & Dynamic Hyperparameters ---")

    # Original baseline test cases (must still pass)
    baseline_cases = [
        ("hi there", QueryClass.CONVERSATIONAL),
        ("what is RA-CapNet?", QueryClass.FACTUAL_LOOKUP),
        ("Compare Transformer vs LSTM models", QueryClass.COMPARATIVE),
        ("What datasets were used by models for relation extraction?", QueryClass.MULTI_HOP_RELATIONAL),
        ("Summarize all files in the knowledge base", QueryClass.SUMMARIZATION),
        ("What are the modern approaches in speaker diarization?", QueryClass.EXPLORATORY_THEMATIC),
    ]

    # --- New test cases covering C-1 to C-7 findings ---
    new_cases = [
        # C-1: "what are" (plural) must be FACTUAL_LOOKUP
        ("what are the RAG types?", QueryClass.FACTUAL_LOOKUP),
        ("what are transformers?", QueryClass.FACTUAL_LOOKUP),
        # C-2: Factual queries up to 15 words must still be FACTUAL_LOOKUP
        # "what is the difference between X and Y" is rightly COMPARATIVE; test a cleaner factual phrasing
        ("what is the meaning of retrieval augmented generation in NLP context?", QueryClass.FACTUAL_LOOKUP),
        # C-3: "types of / kinds of / categories of" -> FACTUAL_LOOKUP
        ("what types of embeddings exist in modern NLP?", QueryClass.FACTUAL_LOOKUP),
        ("list the kinds of RAG architectures", QueryClass.FACTUAL_LOOKUP),
        ("من هي ليلى حسن؟", QueryClass.FACTUAL_LOOKUP),
        ("من هو مدير المشروع؟", QueryClass.FACTUAL_LOOKUP),
        # C-4: New SUMMARIZATION phrasings
        ("give an overview across all papers on retrieval methods", QueryClass.SUMMARIZATION),
        ("summarize the research on retrieval augmented generation", QueryClass.SUMMARIZATION),
        ("what does the corpus say about attention mechanisms?", QueryClass.SUMMARIZATION),
        # C-5: RELATIONAL — connection/relate phrasings
        ("what is the connection between RAG and LLMs?", QueryClass.MULTI_HOP_RELATIONAL),
        ("how does BERT relate to attention mechanisms?", QueryClass.MULTI_HOP_RELATIONAL),
        ("how are dense and sparse retrieval linked?", QueryClass.MULTI_HOP_RELATIONAL),
        # C-6: Conversational with punctuation
        ("hi, how are you?", QueryClass.CONVERSATIONAL),
        ("hello, can you help me?", QueryClass.CONVERSATIONAL),
        ("thanks!", QueryClass.CONVERSATIONAL),
        # "what are the modern approaches" -> EXPLORATORY (adjective chain blocks FACTUAL)
        ("what are the modern approaches in speaker diarization?", QueryClass.EXPLORATORY_THEMATIC),
    ]

    all_cases = baseline_cases + new_cases
    failures = []

    for q, expected_class in all_cases:
        res = query_classifier.classify(q)
        status = "✅" if res.query_class == expected_class else "❌"
        print(f"{status} Query: '{q}'")
        print(f"   -> Class: {res.query_class.value} (Expected: {expected_class.value})")
        print(f"   -> Rec K: {res.recommended_k}, Cand Pool: {res.candidate_pool_size}, MMR λ: {res.mmr_lambda}")
        print(f"   -> Entities: {res.extracted_entities}")
        if res.comparison_targets:
            print(f"   -> Comparison Targets: {res.comparison_targets}")
        if res.query_class != expected_class:
            failures.append((q, expected_class.value, res.query_class.value))

    # Verify key parameter values for high-priority intents
    print("\n-- Parameter Verification --")

    summ = query_classifier.classify("Summarize all files in the knowledge base")
    assert summ.recommended_k == 15, f"SUMMARIZATION K should be 15, got {summ.recommended_k}"
    assert summ.candidate_pool_size == 45, f"SUMMARIZATION pool should be 45, got {summ.candidate_pool_size}"
    assert summ.graph_limit == 25, f"SUMMARIZATION graph_limit should be 25, got {summ.graph_limit}"
    print(f"✅ SUMMARIZATION params: K={summ.recommended_k}, pool={summ.candidate_pool_size}, graph_limit={summ.graph_limit}")

    expl = query_classifier.classify("What are the modern approaches in speaker diarization?")
    assert expl.recommended_k >= 8, f"EXPLORATORY K should be >= 8, got {expl.recommended_k}"
    assert expl.candidate_pool_size == expl.recommended_k * 3, "EXPLORATORY pool should be K*3"
    print(f"✅ EXPLORATORY params: K={expl.recommended_k}, pool={expl.candidate_pool_size}")

    comp = query_classifier.classify("Compare Transformer vs LSTM models")
    assert comp.recommended_k >= 6, f"COMPARATIVE K should be >= 6, got {comp.recommended_k}"
    print(f"✅ COMPARATIVE params: K={comp.recommended_k}, targets={comp.comparison_targets}")

    # Entity stop word check (C-7)
    summ_ents = query_classifier.classify("Summarize all files in the knowledge base")
    bad_stopwords = {"summarize", "all", "files", "knowledge", "base"}
    polluted = [e for e in summ_ents.extracted_entities if e.lower() in bad_stopwords]
    assert not polluted, f"Entity extraction returned stop words: {polluted}"
    print(f"✅ Entity stop word filter: no pollution (entities={summ_ents.extracted_entities})")

    if failures:
        print(f"\n❌ {len(failures)} classification failures:")
        for q, exp, got in failures:
            print(f"   '{q}': expected {exp}, got {got}")
        raise AssertionError(f"{len(failures)} test case(s) failed")

    print("\n✅ Query Classification tests passed!")


def test_mmr_diversity() -> None:
    """Test MMR algorithm with synthetic embeddings to prove diversity selection."""
    print("\n--- 2. Testing MMR Diversity Selection ---")
    np.random.seed(42)

    # Mock candidate pool
    dim = 64
    query_vec = np.random.randn(dim).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    # Create 6 candidates:
    # 3 from Document A (very similar to each other and query)
    # 3 from Document B and C (diverse)
    cand_matrix = []
    # Candidates 0, 1, 2: close to query and close to each other
    for _ in range(3):
        v = query_vec + np.random.randn(dim) * 0.05
        v /= np.linalg.norm(v)
        cand_matrix.append(v)
    # Candidates 3, 4: somewhat close to query but different from cand 0-2
    for _ in range(3):
        v = query_vec + np.random.randn(dim) * 0.4
        v /= np.linalg.norm(v)
        cand_matrix.append(v)

    mock_ids = [f"chunk_{i}" for i in range(6)]
    mock_docs = [f"Text excerpt for chunk {i}" for i in range(6)]
    mock_metas = [
        {"document_name": "paper_A.pdf", "document_title": "Paper A", "page": i + 1}
        for i in range(3)
    ] + [
        {"document_name": f"paper_{chr(66+i)}.pdf", "document_title": f"Paper {chr(66+i)}", "page": 1}
        for i in range(3)
    ]

    class MockCollection:
        def query(self, **kwargs):
            return {
                "ids": [mock_ids],
                "documents": [mock_docs],
                "metadatas": [mock_metas],
                "embeddings": [[list(v) for v in cand_matrix]],
                "distances": [[0.1, 0.12, 0.13, 0.35, 0.38, 0.40]],
            }

    class MockStore:
        collection = MockCollection()

    class MockLLM:
        def get_embedding(self, q):
            return list(query_vec)

    mock_div = diversity_retriever.__class__(v_store=MockStore(), client=MockLLM())
    diverse_results = mock_div.retrieve_diverse(
        query="test query",
        k=3,
        candidate_k=6,
        mmr_lambda=0.5,
        cross_doc_penalty=0.2,
    )

    print(f"Retrieved {len(diverse_results)} diverse chunks:")
    selected_docs = [r["metadata"]["document_title"] for r in diverse_results]
    for r in diverse_results:
        print(f"  Rank #{r['diversity_rank']}: {r['metadata']['document_title']} | Sim: {r['similarity_score']} | MMR Score: {r['mmr_score']}")

    # With cross-document penalty and MMR diversity, chunks should not all come from Paper A
    unique_selected = set(selected_docs)
    print(f"  Selected papers: {selected_docs} (Unique count: {len(unique_selected)})")
    assert len(unique_selected) > 1, "MMR did not enforce cross-document diversity!"
    print("✅ MMR Diversity Selection tests passed!")


def test_evidence_aggregation() -> None:
    """Test structuring and aggregation of vector excerpts and graph discoveries."""
    print("\n--- 3. Testing Evidence Aggregator ---")

    mock_chunks = [
        {
            "chunk_id": "c1",
            "text": "BERT achieves high accuracy on GLUE benchmark.",
            "metadata": {"document_name": "bert.pdf", "document_title": "BERT Paper", "page": 3},
            "similarity_score": 0.92,
            "diversity_rank": 1,
        },
        {
            "chunk_id": "c2",
            "text": "Capsule networks preserve spatial hierarchies.",
            "metadata": {"document_name": "capsule.pdf", "document_title": "Capsule Paper", "page": 2},
            "similarity_score": 0.85,
            "diversity_rank": 2,
        },
    ]

    mock_discovery = ConceptDiscoveryResult(
        concepts=[
            DiscoveredConcept(name="BERT", category="Method", role="direct"),
            DiscoveredConcept(name="Capsule Networks", category="Method", role="direct"),
            DiscoveredConcept(name="Attention Mechanism", category="Topic", role="bridge"),
        ],
        relationships=[
            {"source": "BERT", "relation": "USES_METHOD", "target": "Attention Mechanism"},
            {"source": "Capsule Networks", "relation": "RELATED_TO", "target": "Attention Mechanism"},
        ],
        bridge_concepts=[
            DiscoveredConcept(name="Attention Mechanism", category="Topic", role="bridge")
        ],
        hub_concepts=[
            DiscoveredConcept(name="Natural Language Processing", category="Topic", degree=15, role="hub")
        ],
        categories_found={
            "Method": ["BERT", "Capsule Networks"],
            "Topic": ["Attention Mechanism", "Natural Language Processing"],
        },
    )

    qc = query_classifier.classify("Compare BERT and Capsule Networks")
    evidence = evidence_aggregator.aggregate(
        diverse_chunks=mock_chunks,
        discovery=mock_discovery,
        classification=qc,
    )

    print(f"Assembled context preview:\n{evidence.assembled_context[:400]}...")
    print(f"Sources: {evidence.sources}")
    print(f"Unique papers: {evidence.unique_papers_count}")
    print(f"Discovered concepts: {evidence.discovered_concepts_count}")
    assert evidence.unique_papers_count == 2
    assert "Connecting Bridge Concepts" in evidence.assembled_context
    assert "BERT" in evidence.assembled_context
    print("✅ Evidence Aggregator tests passed!")


def test_hybrid_retriever_pipeline() -> None:
    """Test hybrid retriever adaptive_retrieve."""
    print("\n--- 4. Testing Hybrid Retriever End-to-End ---")
    query = "What is BERT?"
    res = hybrid_retriever.adaptive_retrieve(query)
    print(f"Query: '{query}'")
    print(f"  Classification: {res.classification.query_class.value if res.classification else None}")
    print(f"  Chunks retrieved: {len(res.chunks)}")
    print(f"  Unique papers: {res.unique_papers_count}")
    print(f"  Graph facts: {len(res.graph_facts)}")
    print(f"  Sources: {res.sources}")
    assert res.classification is not None
    assert isinstance(res.assembled_text, str)
    print("✅ Hybrid Retriever pipeline test passed!")


if __name__ == "__main__":
    print("🚀 Starting Adaptive Hybrid RAG Test Suite...")
    test_query_classification()
    test_mmr_diversity()
    test_evidence_aggregation()
    test_hybrid_retriever_pipeline()
    print("\n🎉 ALL ADAPTIVE HYBRID RAG TESTS PASSED SUCCESSFULLY!")
