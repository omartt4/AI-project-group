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
        script_chars = arabic_chars + latin_chars
        
        if script_chars == 0:
            return LanguageDetection(language='unknown', confidence=0.0, detected_script='none')

        # Compare only letters from the supported scripts; punctuation and digits
        # must not dilute the language ratios.
        arabic_ratio = arabic_chars / script_chars
        latin_ratio = latin_chars / script_chars

        # Check for Arabic stopwords for better confidence
        arabic_word_count = 0
        words = text.split()
        for word in words:
            if word.lower() in self.ARABIC_STOPWORDS:
                arabic_word_count += 1
        
        # Determine language
        minority_ratio = min(arabic_ratio, latin_ratio)
        if arabic_chars and latin_chars and minority_ratio >= 0.2:
            return LanguageDetection(
                language='mixed',
                confidence=max(arabic_ratio, latin_ratio),
                detected_script='mixed'
            )
        if arabic_ratio >= 0.8:
            confidence = min(arabic_ratio + (0.1 if arabic_word_count else 0.0), 1.0)
            return LanguageDetection(
                language='ar',
                confidence=confidence,
                detected_script='arabic'
            )
        elif latin_ratio >= 0.8:
            confidence = latin_ratio
            return LanguageDetection(
                language='en',
                confidence=confidence,
                detected_script='latin'
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
            # Use word counts rather than character counts so long English
            # technical terms do not override an Arabic question (or vice
            # versa). If tied, follow the script of the first meaningful word.
            tokens = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[A-Za-z]+", query)
            arabic_words = sum(bool(self.ARABIC_PATTERN.search(token)) for token in tokens)
            latin_words = sum(bool(self.LATIN_PATTERN.search(token)) for token in tokens)
            first_is_arabic = bool(tokens and self.ARABIC_PATTERN.search(tokens[0]))
            if arabic_words > latin_words or (arabic_words == latin_words and first_is_arabic):
                return "Respond in Arabic (العربية) as the query is primarily in Arabic."
            else:
                return "Respond in English as the query is primarily in English."
        else:
            # Default to English
            return "Respond in English."


language_detector = LanguageDetector()
