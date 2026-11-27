"""
Language Detection module for Multilingual RAG.

Detects the language of user queries to enable:
- Language-aware retrieval strategies
- Cross-lingual query expansion
- Appropriate LLM response language
"""

import logging
import re
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """Language detection result."""
    language: str  # 'ar', 'en', 'mixed', 'unknown'
    confidence: float
    detected_script: str


class LanguageDetector:
    """
    Detects language from text using script-based heuristics.
    Fast, lightweight, no external dependencies.
    """

    # Arabic character ranges (Unicode)
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    
    # Latin script (English and other Latin-based languages)
    LATIN_PATTERN = re.compile(r'[a-zA-Z]')
    
    # Common Arabic words for better detection
    ARABIC_STOPWORDS = {
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
        'ما', 'من', 'هل', 'كيف', 'لماذا', 'أين', 'متى', 'من', 'أين',
        'الذي', 'التي', 'الذين', 'اللاتي', 'كان', 'كانت', 'يكون',
        'ليس', 'ليست', 'لا', 'لم', 'لن', 'إن', 'أن', 'أو', 'ثم',
    }

    def detect(self, text: str) -> LanguageDetection:
        """
        Detect the primary language of the input text.

        Returns:
            LanguageDetection with language code and confidence
        """
        if not text or not text.strip():
            return LanguageDetection(language='unknown', confidence=0.0, detected_script='none')

        text = text.strip()
        
        # Count characters by script
        arabic_chars = len(self.ARABIC_PATTERN.findall(text))
        latin_chars = len(self.LATIN_PATTERN.findall(text))
        total_chars = len(text.replace(' ', ''))  # Exclude spaces
        
        if total_chars == 0:
            return LanguageDetection(language='unknown', confidence=0.0, detected_script='none')

        # Calculate proportions
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
        latin_ratio = latin_chars / total_chars if total_chars > 0 else 0

        # Check for Arabic stopwords for better confidence
        arabic_word_count = 0
        words = text.split()
        for word in words:
            if word.lower() in self.ARABIC_STOPWORDS:
                arabic_word_count += 1
        
        # Boost Arabic confidence if stopwords present
        if arabic_word_count > 0:
            arabic_ratio = min(arabic_ratio + 0.2, 1.0)

        # Determine language
        if arabic_ratio > 0.7:
            confidence = arabic_ratio
            return LanguageDetection(
                language='ar',
                confidence=confidence,
                detected_script='arabic'
            )
        elif latin_ratio > 0.7:
            confidence = latin_ratio
            return LanguageDetection(
                language='en',
                confidence=confidence,
                detected_script='latin'
            )
        elif arabic_ratio > 0.3 and latin_ratio > 0.3:
            # Mixed language
            confidence = max(arabic_ratio, latin_ratio)
            return LanguageDetection(
                language='mixed',
                confidence=confidence,
                detected_script='mixed'
            )
        else:
            # Low confidence - default to English for scientific queries
            return LanguageDetection(
                language='en',
                confidence=0.5,
                detected_script='unknown'
            )

    def get_response_language_instruction(
        self, 
        detection: LanguageDetection,
        query: str
    ) -> str:
        """
        Get language instruction for LLM based on detected language.
        
        Returns:
            String instruction for the LLM about response language
        """
        if detection.language == 'ar':
            return "Respond in Arabic (العربية)."
        elif detection.language == 'mixed':
            # For mixed queries, respond in the dominant language
            arabic_chars = len(self.ARABIC_PATTERN.findall(query))
            latin_chars = len(self.LATIN_PATTERN.findall(query))
            if arabic_chars > latin_chars:
                return "Respond in Arabic (العربية) as the query is primarily in Arabic."
            else:
                return "Respond in English as the query is primarily in English."
        else:
            # Default to English
            return "Respond in English."


language_detector = LanguageDetector()
