"""
Unit tests for Evidence Quality Assessment module.

Tests the assess_evidence_quality function for various retrieval scenarios.
"""

import pytest
from graph_rag.retriever.query_engine import assess_evidence_quality, QueryClassification, QueryClass


class TestEvidenceQuality:
    """Test suite for evidence quality assessment."""

    def test_empty_chunks_insufficient(self):
        """Test that empty chunks are marked as insufficient."""
        is_insufficient, quality = assess_evidence_quality(
            chunks=[],
            context_text="",
            classification=None,
        )
        
        assert is_insufficient is True
        assert quality == 0.0

    def test_sufficient_chunks_high_quality(self):
        """Test that sufficient chunks with good similarity pass quality check."""
        chunks = [
            {"similarity_score": 0.85, "text": "Relevant content" * 20},
            {"similarity_score": 0.90, "text": "More relevant content" * 20},
            {"similarity_score": 0.88, "text": "Additional relevant content" * 20},
        ]
        context_text = " ".join([c["text"] for c in chunks])
        
        is_insufficient, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=None,
        )
        
        assert is_insufficient is False
        assert quality > 0.5

    def test_low_similarity_insufficient(self):
        """Test that low similarity scores trigger insufficient evidence."""
        chunks = [
            {"similarity_score": 0.2, "text": "Irrelevant content" * 20},
            {"similarity_score": 0.25, "text": "More irrelevant content" * 20},
        ]
        context_text = " ".join([c["text"] for c in chunks])
        
        is_insufficient, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=None,
        )
        
        assert is_insufficient is True
        assert quality < 0.4

    def test_short_context_insufficient(self):
        """Test that very short context is marked as insufficient."""
        chunks = [
            {"similarity_score": 0.8, "text": "Short"},
        ]
        context_text = "Short"
        
        is_insufficient, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=None,
        )
        
        assert is_insufficient is True

    def test_factual_query_stricter_threshold(self):
        """Test that factual queries require higher quality evidence."""
        chunks = [
            {"similarity_score": 0.5, "text": "Moderately relevant content" * 20},
            {"similarity_score": 0.55, "text": "More moderately relevant content" * 20},
        ]
        context_text = " ".join([c["text"] for c in chunks])
        
        classification = QueryClassification(
            query_class=QueryClass.FACTUAL_LOOKUP,
            confidence=0.9,
            recommended_k=3,
        )
        
        is_insufficient, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=classification,
        )
        
        # Factual queries should be stricter
        assert is_insufficient is True or quality < 0.6

    def test_exploratory_query_lenient_threshold(self):
        """Test that exploratory queries are more lenient."""
        chunks = [
            {"similarity_score": 0.5, "text": "Moderately relevant content" * 20},
            {"similarity_score": 0.55, "text": "More moderately relevant content" * 20},
        ]
        context_text = " ".join([c["text"] for c in chunks])
        
        classification = QueryClassification(
            query_class=QueryClass.EXPLORATORY_THEMATIC,
            confidence=0.8,
            recommended_k=8,
        )
        
        is_insufficient, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=classification,
        )
        
        # Exploratory queries should be more lenient
        assert quality >= 0.4 or is_insufficient is False

    def test_quality_score_range(self):
        """Test that quality score is always between 0 and 1."""
        chunks = [
            {"similarity_score": 0.5, "text": "Content" * 20},
        ]
        context_text = chunks[0]["text"]
        
        _, quality = assess_evidence_quality(
            chunks=chunks,
            context_text=context_text,
            classification=None,
        )
        
        assert 0.0 <= quality <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
