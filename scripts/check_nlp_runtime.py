"""Validate local OCR and spaCy runtimes without calling cloud APIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.ingestion.ocr import create_ocr_provider
from graph_rag.ingestion.spacy_candidates import spacy_candidate_extractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-spacy", action="store_true")
    parser.add_argument("--require-arabic-dependencies", action="store_true")
    parser.add_argument(
        "--require-ocr",
        choices=["paddle", "surya", "tesseract", "auto"],
    )
    args = parser.parse_args()

    spacy_ok = spacy_candidate_extractor.is_available()
    print(f"spaCy runtime: {'OK' if spacy_ok else 'UNAVAILABLE (lexical fallback)'}")
    if args.require_spacy and not spacy_ok:
        return 1

    if args.require_arabic_dependencies:
        analysis = spacy_candidate_extractor.extract(
            "يدير أحمد شركة تقنية كبيرة في القاهرة.",
            language="ar",
        )
        dependencies = analysis.get("dependencies") or []
        parser_model = analysis.get("parser_model") or ""
        arabic_dependencies_ok = bool(dependencies) and parser_model == "stanza_ar_ud"
        print(
            "Arabic dependencies: "
            f"{'OK' if arabic_dependencies_ok else 'UNAVAILABLE'} "
            f"(model={parser_model or 'none'}, rows={len(dependencies)})"
        )
        if not arabic_dependencies_ok:
            return 1

    if args.require_ocr:
        provider = create_ocr_provider(args.require_ocr)
        ocr_ok = provider.is_available()
        print(f"OCR {args.require_ocr}: {'OK' if ocr_ok else 'UNAVAILABLE'}")
        if not ocr_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
