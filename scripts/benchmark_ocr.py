"""Benchmark local OCR providers with CER, WER, latency, boxes, and tables.

The benchmark never calls OpenRouter. Each JSONL row must contain a manually
verified ``ground_truth`` plus either ``image`` or ``pdf`` and ``page_number``.

Example:
    uv run python scripts/benchmark_ocr.py \
        --manifest evaluation_datasets/ocr/manifest.jsonl \
        --providers paddle tesseract surya
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.ingestion.ocr import create_ocr_provider


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").split())


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Return Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_item in enumerate(reference, 1):
        current = [row_index]
        for column_index, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_text = _normalized_text(reference)
    hypothesis_text = _normalized_text(hypothesis)
    return _edit_distance(reference_text, hypothesis_text) / max(1, len(reference_text))


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _normalized_text(reference).split()
    hypothesis_words = _normalized_text(hypothesis).split()
    return _edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))


@dataclass
class BenchmarkRow:
    sample_id: str
    provider: str
    language: str
    kind: str
    page_number: int
    status: str
    cer: float | None
    wer: float | None
    latency_ms: int
    confidence: float | None
    bounding_box_count: int
    expected_tables: int
    detected_tables: int
    recognized_text: str
    error: str = ""


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if not str(row.get("ground_truth") or "").strip():
                raise ValueError(
                    f"Manifest line {line_number} has no verified ground_truth."
                )
            if not row.get("image") and not row.get("pdf"):
                raise ValueError(
                    f"Manifest line {line_number} needs image or pdf."
                )
            rows.append(row)
    if not rows:
        raise ValueError("OCR benchmark manifest contains no samples.")
    return rows


def _sample_bytes(row: dict[str, Any], manifest_dir: Path, dpi: int) -> bytes:
    if row.get("image"):
        return (manifest_dir / str(row["image"])).resolve().read_bytes()
    import pymupdf

    pdf_path = (manifest_dir / str(row["pdf"])).resolve()
    page_number = int(row.get("page_number") or 1)
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Invalid page_number={page_number} for {pdf_path.name}."
            )
        return document[page_number - 1].get_pixmap(
            dpi=dpi,
            alpha=False,
        ).tobytes("png")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "ocr"


def _write_results(rows: Iterable[BenchmarkRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    (output_dir / "ocr_benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(BenchmarkRow.__dataclass_fields__)
    with (output_dir / "ocr_benchmark_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)

    summaries = []
    for provider in sorted({row.provider for row in rows}):
        successful = [
            row for row in rows if row.provider == provider and row.status == "ok"
        ]
        summaries.append(
            {
                "provider": provider,
                "samples": len(successful),
                "mean_cer": (
                    sum(row.cer or 0.0 for row in successful) / len(successful)
                    if successful
                    else None
                ),
                "mean_wer": (
                    sum(row.wer or 0.0 for row in successful) / len(successful)
                    if successful
                    else None
                ),
                "mean_latency_ms": (
                    sum(row.latency_ms for row in successful) / len(successful)
                    if successful
                    else None
                ),
                "boxes": sum(row.bounding_box_count for row in successful),
                "tables_detected": sum(row.detected_tables for row in successful),
            }
        )
    (output_dir / "ocr_benchmark_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["paddle", "tesseract", "surya"],
        choices=["paddle", "tesseract", "surya"],
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ocr_benchmark",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    samples = _read_manifest(manifest)
    results: list[BenchmarkRow] = []
    for provider_name in args.providers:
        provider = create_ocr_provider(provider_name)
        if not provider.is_available():
            print(f"[SKIP] {provider_name}: runtime unavailable")
            for sample in samples:
                results.append(
                    BenchmarkRow(
                        sample_id=str(sample.get("id") or "unknown"),
                        provider=provider_name,
                        language=str(sample.get("language") or "und"),
                        kind=str(sample.get("kind") or "unknown"),
                        page_number=int(sample.get("page_number") or 1),
                        status="unavailable",
                        cer=None,
                        wer=None,
                        latency_ms=0,
                        confidence=None,
                        bounding_box_count=0,
                        expected_tables=int(sample.get("expected_tables") or 0),
                        detected_tables=0,
                        recognized_text="",
                        error="provider runtime unavailable",
                    )
                )
            continue
        for sample in samples:
            sample_id = str(sample.get("id") or _safe_name(provider_name))
            started = time.perf_counter()
            try:
                image_bytes = _sample_bytes(sample, manifest.parent, args.dpi)
                output = provider.extract(
                    image_bytes,
                    language=str(sample.get("language") or "eng+ara"),
                )
                latency_ms = round((time.perf_counter() - started) * 1000)
                truth = str(sample["ground_truth"])
                results.append(
                    BenchmarkRow(
                        sample_id=sample_id,
                        provider=provider_name,
                        language=str(sample.get("language") or "und"),
                        kind=str(sample.get("kind") or "unknown"),
                        page_number=int(sample.get("page_number") or 1),
                        status="ok",
                        cer=character_error_rate(truth, output.text),
                        wer=word_error_rate(truth, output.text),
                        latency_ms=latency_ms,
                        confidence=output.confidence,
                        bounding_box_count=len(output.bounding_boxes),
                        expected_tables=int(sample.get("expected_tables") or 0),
                        detected_tables=len(output.tables),
                        recognized_text=output.text,
                    )
                )
                print(f"[OK] {provider_name}: {sample_id} ({latency_ms} ms)")
            except Exception as exc:
                results.append(
                    BenchmarkRow(
                        sample_id=sample_id,
                        provider=provider_name,
                        language=str(sample.get("language") or "und"),
                        kind=str(sample.get("kind") or "unknown"),
                        page_number=int(sample.get("page_number") or 1),
                        status="failed",
                        cer=None,
                        wer=None,
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        confidence=None,
                        bounding_box_count=0,
                        expected_tables=int(sample.get("expected_tables") or 0),
                        detected_tables=0,
                        recognized_text="",
                        error=str(exc),
                    )
                )
                print(f"[FAIL] {provider_name}: {sample_id}: {exc}")

    _write_results(results, args.output_dir.resolve())
    return 0 if any(row.status == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
