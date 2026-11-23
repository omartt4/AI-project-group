"""Lightweight script-based language detection for extracted text."""

import unicodedata


def detect_language(text: str) -> str:
    has_arabic = False
    has_latin = False
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        has_arabic |= "ARABIC" in name
        has_latin |= "LATIN" in name
        if has_arabic and has_latin:
            return "mixed"
    if has_arabic:
        return "ar"
    if has_latin:
        return "en"
    return "und"
