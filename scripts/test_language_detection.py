"""
Simple Language Detection Test.

Tests the language detector without requiring full system initialization.
This avoids PyMuPDF import issues.

Usage:
    uv run python scripts/test_language_detection.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from graph_rag.retriever.language_detector import language_detector
    
    test_queries = [
        ("What is deep learning?", "en"),
        ("ما هو التعلم العميق؟", "ar"),
        ("What is التعلم العميق؟", "mixed"),
        ("ما هو deep learning؟", "mixed"),
    ]

    print("Language Detection Test")
    print("=" * 60)

    for query, expected_lang in test_queries:
        result = language_detector.detect(query)
        status = "✓" if result.language == expected_lang else "✗"
        print(f"{status} Query: {query}")
        print(f"   Detected: {result.language} (expected: {expected_lang})")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Script: {result.detected_script}")
        print()

    print("Test completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
