"""
Diversity-Aware Retriever module using Maximal Marginal Relevance (MMR).

Eliminates redundancy in retrieved passages and balances:
1. Relevance to the user's query
2. Novelty/diversity with respect to already selected passages
3. Cross-document coverage across research papers
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from graph_rag.embeddings.vector_store import vector_store
from graph_rag.llm.ollama_client import ollama_client

logger = logging.getLogger(__name__)


class DiversityRetriever:
    """Retrieves diverse, non-redundant passages using Maximal Marginal Relevance (MMR)."""

    def __init__(self, v_store=None, client=None) -> None:
        self.vector_store = v_store or vector_store
        self.llm_client = client or ollama_client

    def retrieve_diverse(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
        mmr_lambda: float = 0.6,
        cross_doc_penalty: float = 0.15,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Diversity-Aware Vector Retrieval with MMR.

        Args:
            query: The user query string.
            k: Final number of diverse chunks to return.
            candidate_k: Number of initial candidates to pull from ChromaDB (candidate pool).
            mmr_lambda: Weight for relevance vs diversity (1.0 = pure relevance, 0.0 = pure diversity).
            cross_doc_penalty: Penalty factor applied to candidates from documents already in selection.
            filter_metadata: Optional ChromaDB metadata filter.

        Returns:
            Ranked list of diverse chunks with similarity and diversity scores.
        """
        # Ensure candidate pool is large enough for diversity selection
        candidate_k = max(candidate_k, k * 2)

        # 1. Generate query embedding
        try:
            query_embedding = self.llm_client.get_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return []

        # 2. Fetch candidate pool with embeddings from ChromaDB
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            logger.warning(f"ChromaDB candidate query failed: {e}")
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        embeddings = results["embeddings"][0]
        distances = results.get("distances", [[]])[0]

        n_candidates = len(ids)
        if n_candidates == 0:
            return []

        # If candidates are fewer than or equal to k, format and return them directly
        if n_candidates <= k:
            formatted = []
            for i in range(n_candidates):
                dist = distances[i] if i < len(distances) else 0.0
                formatted.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "similarity_score": round(max(0.0, 1.0 - dist), 4),
                    "mmr_score": round(max(0.0, 1.0 - dist), 4),
                    "diversity_rank": i + 1,
                })
            return formatted

        # Convert to numpy arrays for fast vectorized MMR calculation
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-8:
            q_vec = q_vec / q_norm

        cand_matrix = np.array(embeddings, dtype=np.float32)
        cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        cand_norms[cand_norms < 1e-8] = 1.0
        cand_matrix_norm = cand_matrix / cand_norms

        # Compute cosine similarities between candidates and query: dot(C, q)
        sim_to_query = np.dot(cand_matrix_norm, q_vec)

        # 3. Iterative MMR Selection
        selected_indices: List[int] = []
        unselected_indices = list(range(n_candidates))
        selected_doc_counts: Dict[str, int] = {}
        mmr_scores: Dict[int, float] = {}

        # First item is the candidate with the highest similarity to query
        first_idx = int(np.argmax(sim_to_query))
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        mmr_scores[first_idx] = float(sim_to_query[first_idx])

        first_doc = metadatas[first_idx].get("document_name", "unknown")
        selected_doc_counts[first_doc] = 1

        # Iteratively pick remaining items up to k
        for _ in range(1, min(k, n_candidates)):
            best_idx = None
            best_mmr = -float("inf")

            selected_matrix = cand_matrix_norm[selected_indices]  # Shape: (num_selected, dim)

            for cand_idx in unselected_indices:
                cand_vec = cand_matrix_norm[cand_idx]
                relevance = float(sim_to_query[cand_idx])

                # Max similarity to any already-selected chunk
                pairwise_sims = np.dot(selected_matrix, cand_vec)
                max_sim_to_selected = float(np.max(pairwise_sims))

                # Base MMR score formula: λ * Sim(c, q) - (1 - λ) * max(Sim(c, s))
                mmr_val = (mmr_lambda * relevance) - ((1.0 - mmr_lambda) * max_sim_to_selected)

                # Cross-document coverage bonus/penalty:
                # If this document is already selected, penalize slightly to promote diverse sources
                doc_name = metadatas[cand_idx].get("document_name", "unknown")
                times_doc_selected = selected_doc_counts.get(doc_name, 0)
                if times_doc_selected > 0:
                    mmr_val -= cross_doc_penalty * times_doc_selected

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = cand_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
                mmr_scores[best_idx] = best_mmr
                doc_name = metadatas[best_idx].get("document_name", "unknown")
                selected_doc_counts[doc_name] = selected_doc_counts.get(doc_name, 0) + 1

        # 4. Assemble final diverse results
        diverse_results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(selected_indices, start=1):
            dist = distances[idx] if idx < len(distances) else 0.0
            sim_score = max(0.0, 1.0 - dist)
            diverse_results.append({
                "chunk_id": ids[idx],
                "text": documents[idx],
                "metadata": metadatas[idx],
                "similarity_score": round(float(sim_score), 4),
                "mmr_score": round(float(mmr_scores.get(idx, sim_score)), 4),
                "diversity_rank": rank,
            })

        logger.debug(
            "MMR Diversity Selection completed: candidates=%d, returned=%d, unique_docs=%d",
            n_candidates,
            len(diverse_results),
            len(selected_doc_counts),
        )
        return diverse_results


diversity_retriever = DiversityRetriever()

