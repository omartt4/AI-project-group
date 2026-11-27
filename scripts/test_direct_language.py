"""
Direct Language Detection Test.

Tests the language detector module directly without importing other modules.
"""

import re
from typing import Optional
from dataclasses import dataclass

@dataclass
class LanguageDetection:
    language: str
    confidence: float
    detected_script: str

class LanguageDetector:
    def __init__(self):
        self.arabic_stopwords = {
            "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك",
            "التي", "الذي", "الذين", "كان", "كانت", "يكون", "تكون",
            "ما", "هل", "كيف", "لماذا", "أين", "متى", "من", "في",
        }
    
    def detect(self, text: str) -> LanguageDetection:
        text = text.strip()
        if not text:
            return LanguageDetection("unknown", 0.0, "none")
        
        # Check for Arabic script
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = arabic_chars + latin_chars
        
        if total_chars == 0:
            return LanguageDetection("unknown", 0.0, "none")
        
        arabic_ratio = arabic_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        # Check for Arabic stopwords
        words = text.split()
        arabic_word_count = sum(1 for w in words if w.lower() in self.arabic_stopwords)
        
        if arabic_ratio > 0.7:
            return LanguageDetection("ar", 0.9, "arabic")
        elif latin_ratio > 0.7:
            return LanguageDetection("en", 0.9, "latin")
        elif arabic_word_count >= 2:
            return LanguageDetection("ar", 0.8, "arabic")
        elif arabic_ratio > 0.3 and latin_ratio > 0.3:
            return LanguageDetection("mixed", 0.7, "mixed")
        elif arabic_ratio > latin_ratio:
            return LanguageDetection("ar", 0.6, "arabic")
        else:
            return LanguageDetection("en", 0.6, "latin")

# Test
detector = LanguageDetector()

test_queries = [
    ("What is deep learning?", "en"),
    ("ما هو التعلم العميق؟", "ar"),
    ("What is التعلم العميق؟", "mixed"),
    ("ما هو deep learning؟", "mixed"),
]

print("Language Detection Test (Direct)")
print("=" * 60)

for query, expected_lang in test_queries:
    result = detector.detect(query)
    status = "✓" if result.language == expected_lang else "✗"
    print(f"{status} Query: {query}")
    print(f"   Detected: {result.language} (expected: {expected_lang})")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Script: {result.detected_script}")
    print()

print("Test completed!")
