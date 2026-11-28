"""
Unit tests for Language Detector module.

Tests language detection accuracy for Arabic, English, and mixed queries.
"""

import pytest
from graph_rag.retriever.language_detector import language_detector, LanguageDetection


class TestLanguageDetector:
    """Test suite for language detection functionality."""

    def test_detect_arabic_query(self):
        """Test detection of pure Arabic queries."""
        arabic_query = "ما هو التعلم العميق؟"
        result = language_detector.detect(arabic_query)
        
        assert result.language == "ar"
        assert result.confidence > 0.7
        assert result.detected_script == "arabic"

    def test_detect_english_query(self):
        """Test detection of pure English queries."""
        english_query = "What is deep learning?"
        result = language_detector.detect(english_query)
        
        assert result.language == "en"
        assert result.confidence > 0.7
        assert result.detected_script == "latin"

    def test_detect_mixed_query_arabic_dominant(self):
        """Test detection of mixed queries with Arabic dominance."""
        mixed_query = "ما هو deep learning؟"
        result = language_detector.detect(mixed_query)
        
        assert result.language in ["ar", "mixed"]
        assert result.confidence > 0.5

    def test_detect_mixed_query_english_dominant(self):
        """Test detection of mixed queries with English dominance."""
        mixed_query = "What is التعلم العميق؟"
        result = language_detector.detect(mixed_query)
        
        assert result.language in ["en", "mixed"]
        assert result.confidence > 0.5

    def test_detect_empty_query(self):
        """Test detection of empty query."""
        result = language_detector.detect("")
        
        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_detect_arabic_with_stopwords(self):
        """Test Arabic detection with common stopwords."""
        query = "في هذا البحث ندرس كيف يمكن استخدام الشبكات العصبية"
        result = language_detector.detect(query)
        
        assert result.language == "ar"
        assert result.confidence > 0.8  # Higher confidence with stopwords

    def test_get_response_language_instruction_arabic(self):
        """Test language instruction for Arabic."""
        detection = LanguageDetection(language="ar", confidence=0.9, detected_script="arabic")
        instruction = language_detector.get_response_language_instruction(detection, "سؤال")
        
        assert "Arabic" in instruction or "العربية" in instruction

    def test_get_response_language_instruction_english(self):
        """Test language instruction for English."""
        detection = LanguageDetection(language="en", confidence=0.9, detected_script="latin")
        instruction = language_detector.get_response_language_instruction(detection, "question")
        
        assert "English" in instruction

    def test_get_response_language_instruction_mixed_arabic(self):
        """Test language instruction for mixed query (Arabic dominant)."""
        detection = LanguageDetection(language="mixed", confidence=0.7, detected_script="mixed")
        query = "ما هو deep learning؟"
        instruction = language_detector.get_response_language_instruction(detection, query)
        
        assert "Arabic" in instruction or "العربية" in instruction

    def test_get_response_language_instruction_mixed_english(self):
        """Test language instruction for mixed query (English dominant)."""
        detection = LanguageDetection(language="mixed", confidence=0.7, detected_script="mixed")
        query = "What is التعلم العميق؟"
        instruction = language_detector.get_response_language_instruction(detection, query)
        
        assert "English" in instruction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
