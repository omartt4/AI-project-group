"""Replaceable OCR provider interface and local Tesseract implementation."""

from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class OCRResult:
    text: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None


class OCRProvider(Protocol):
    def extract(self, image_bytes: bytes, language: str) -> OCRResult:
        """Recognize text in an image and return provenance when available."""


class TesseractOCRProvider:
    """Run local Tesseract OCR and retain word-level confidence and boxes."""

    @staticmethod
    def _configure_windows_executable(pytesseract_module: Any) -> None:
        """Find a standard Windows install even when it is absent from PATH."""
        configured = str(pytesseract_module.pytesseract.tesseract_cmd or "tesseract")
        if shutil.which(configured) or Path(configured).is_file():
            return
        candidates = [
            os.getenv("TESSERACT_CMD", ""),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(
                Path(os.getenv("LOCALAPPDATA", ""))
                / "Programs"
                / "Tesseract-OCR"
                / "tesseract.exe"
            ),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                pytesseract_module.pytesseract.tesseract_cmd = candidate
                return

    def extract(self, image_bytes: bytes, language: str) -> OCRResult:
        try:
            from PIL import Image
            import pytesseract
            from pytesseract import Output
            from pytesseract.pytesseract import (
                TesseractError,
                TesseractNotFoundError,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional Pillow and pytesseract packages to use OCR."
            ) from exc

        self._configure_windows_executable(pytesseract)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                output_type=Output.DICT,
            )
        except TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract is not installed or is not available on PATH."
            ) from exc
        except TesseractError as exc:
            raise RuntimeError(
                f"Tesseract OCR failed for language configuration {language!r}: {exc}"
            ) from exc

        lines: Dict[tuple[int, int, int], List[str]] = {}
        boxes: List[Dict[str, Any]] = []
        confidences: List[float] = []
        for index, value in enumerate(data.get("text", [])):
            word = str(value).strip()
            if not word:
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                confidence = -1.0
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            lines.setdefault(key, []).append(word)
            box = {
                "text": word,
                "x": int(data["left"][index]),
                "y": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
                "confidence": confidence / 100 if confidence >= 0 else None,
                "coordinate_unit": "pixel",
            }
            boxes.append(box)
            if confidence >= 0:
                confidences.append(confidence / 100)

        text = "\n".join(" ".join(words) for words in lines.values())
        average_confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )
        return OCRResult(
            text=text,
            bounding_boxes=boxes,
            confidence=average_confidence,
        )
