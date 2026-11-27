"""Replaceable local OCR providers for PaddleOCR, Surya, and Tesseract."""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Protocol

from graph_rag.config import settings


logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OCRProvider(Protocol):
    def is_available(self) -> bool:
        """Return whether the provider runtime is available."""

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

    def is_available(self) -> bool:
        """Check if pytesseract and the underlying binary are available."""
        try:
            import pytesseract
            self._configure_windows_executable(pytesseract)
            version = pytesseract.get_tesseract_version()
            return bool(version)
        except Exception:
            return False

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
            metadata={"engine": "tesseract", "language": language},
        )


def _box_from_polygon(polygon: Any) -> Optional[Dict[str, int]]:
    """Convert a four-point polygon or x1/y1/x2/y2 box to pixel geometry."""
    if not isinstance(polygon, (list, tuple)) or not polygon:
        return None
    try:
        if len(polygon) == 4 and all(
            isinstance(value, (int, float)) for value in polygon
        ):
            x1, y1, x2, y2 = [float(value) for value in polygon]
        else:
            points = [point for point in polygon if len(point) >= 2]
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return {
            "x": int(round(x1)),
            "y": int(round(y1)),
            "width": max(0, int(round(x2 - x1))),
            "height": max(0, int(round(y2 - y1))),
        }
    except (TypeError, ValueError):
        return None


class PaddleOCRProvider:
    """Run PaddleOCR locally and retain recognized text, confidence, and boxes."""

    def __init__(self, language: Optional[str] = None) -> None:
        self.language = language or settings.PADDLE_OCR_LANGUAGE
        self.service_url = settings.PADDLE_OCR_URL
        self._pipelines: Dict[str, Any] = {}

    def is_available(self) -> bool:
        if self.service_url:
            try:
                import httpx

                response = httpx.get(
                    f"{self.service_url}/health",
                    timeout=3.0,
                )
                return response.status_code == 200
            except Exception:
                return False
        try:
            import paddleocr  # noqa: F401
            import paddle  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _language_for(requested: str, configured: str) -> str:
        value = (requested or configured or "en").casefold()
        if any(code in value for code in ("ara", "ar", "fas", "urd")):
            return "ar"
        return configured or "en"

    def _pipeline(self, language: str) -> Any:
        if language in self._pipelines:
            return self._pipelines[language]
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR is unavailable. Install the paddle-ocr project extra "
                "or run the Docker image."
            ) from exc
        options = {
            "lang": language,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        try:
            pipeline = PaddleOCR(**options)
        except TypeError:
            # Compatibility with PaddleOCR 2.x while the project targets 3.x.
            pipeline = PaddleOCR(lang=language, use_angle_cls=False)
        self._pipelines[language] = pipeline
        return pipeline

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            payload = value
        else:
            payload = getattr(value, "json", None)
            payload = payload() if callable(payload) else payload
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if not isinstance(payload, dict) and hasattr(value, "to_dict"):
                payload = value.to_dict()
        if not isinstance(payload, dict):
            return {}
        return payload.get("res", payload) if isinstance(payload.get("res"), dict) else payload

    @classmethod
    def _parse_results(cls, rows: Any) -> OCRResult:
        texts: List[str] = []
        boxes: List[Dict[str, Any]] = []
        confidences: List[float] = []
        tables: List[Dict[str, Any]] = []
        if rows is None:
            rows = []
        if not isinstance(rows, (list, tuple)):
            rows = list(rows)

        for row in rows:
            payload = cls._mapping(row)
            if payload:
                rec_texts = payload.get("rec_texts") or payload.get("texts") or []
                rec_scores = payload.get("rec_scores") or payload.get("scores") or []
                rec_boxes = (
                    payload.get("rec_polys")
                    or payload.get("dt_polys")
                    or payload.get("rec_boxes")
                    or []
                )
                for index, text in enumerate(rec_texts):
                    clean = str(text).strip()
                    if not clean:
                        continue
                    score = (
                        float(rec_scores[index])
                        if index < len(rec_scores)
                        else None
                    )
                    box = (
                        _box_from_polygon(rec_boxes[index])
                        if index < len(rec_boxes)
                        else None
                    )
                    texts.append(clean)
                    if score is not None:
                        confidences.append(score)
                    boxes.append(
                        {
                            "text": clean,
                            **(box or {}),
                            "confidence": score,
                            "coordinate_unit": "pixel",
                        }
                    )
                table_rows = payload.get("table_res_list") or payload.get("tables") or []
                if isinstance(table_rows, list):
                    tables.extend(
                        item if isinstance(item, dict) else {"value": str(item)}
                        for item in table_rows
                    )
                continue

            # PaddleOCR 2.x returns [[polygon, (text, confidence)], ...].
            legacy_rows = row if isinstance(row, (list, tuple)) else []
            for item in legacy_rows:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                recognition = item[1]
                if not isinstance(recognition, (list, tuple)) or not recognition:
                    continue
                clean = str(recognition[0]).strip()
                if not clean:
                    continue
                score = float(recognition[1]) if len(recognition) > 1 else None
                box = _box_from_polygon(item[0])
                texts.append(clean)
                if score is not None:
                    confidences.append(score)
                boxes.append(
                    {
                        "text": clean,
                        **(box or {}),
                        "confidence": score,
                        "coordinate_unit": "pixel",
                    }
                )

        return OCRResult(
            text="\n".join(texts),
            bounding_boxes=boxes,
            confidence=(sum(confidences) / len(confidences) if confidences else None),
            tables=tables,
            metadata={"engine": "paddleocr"},
        )

    def extract(self, image_bytes: bytes, language: str) -> OCRResult:
        requested_language = self._language_for(language, self.language)
        if self.service_url:
            try:
                import httpx

                response = httpx.post(
                    f"{self.service_url}/ocr",
                    params={"language": requested_language},
                    content=image_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=settings.PADDLE_OCR_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                return OCRResult(
                    text=str(payload.get("text") or ""),
                    bounding_boxes=list(payload.get("bounding_boxes") or []),
                    confidence=payload.get("confidence"),
                    tables=list(payload.get("tables") or []),
                    metadata={
                        **dict(payload.get("metadata") or {}),
                        "engine": "paddleocr-service",
                        "language": requested_language,
                    },
                )
            except Exception as exc:
                raise RuntimeError(f"PaddleOCR service request failed: {exc}") from exc
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("PaddleOCR requires Pillow and NumPy.") from exc
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pipeline = self._pipeline(requested_language)
        if hasattr(pipeline, "predict"):
            rows = pipeline.predict(np.asarray(image))
        else:
            rows = pipeline.ocr(np.asarray(image), cls=False)
        result = self._parse_results(rows)
        result.metadata.update({"language": requested_language})
        return result


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _html_text(value: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(value)
    return " ".join(parser.parts)


class SuryaOCRProvider:
    """Use an installed Surya CLI/backend and parse its JSON provenance."""

    def __init__(
        self,
        command: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.command = command or settings.SURYA_OCR_COMMAND
        self.timeout_seconds = timeout_seconds or settings.SURYA_OCR_TIMEOUT_SECONDS

    def is_available(self) -> bool:
        return bool(shutil.which(self.command) or Path(self.command).is_file())

    @staticmethod
    def _blocks(payload: Any) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            if "html" in payload or "text" in payload:
                blocks.append(payload)
            for value in payload.values():
                blocks.extend(SuryaOCRProvider._blocks(value))
        elif isinstance(payload, list):
            for value in payload:
                blocks.extend(SuryaOCRProvider._blocks(value))
        return blocks

    def extract(self, image_bytes: bytes, language: str) -> OCRResult:
        if not self.is_available():
            raise RuntimeError(
                "Surya OCR CLI is unavailable. Configure SURYA_OCR_COMMAND or "
                "start its Docker inference backend."
            )
        with tempfile.TemporaryDirectory(prefix="graph-rag-surya-") as directory:
            image_path = Path(directory) / "page.png"
            image_path.write_bytes(image_bytes)
            completed = subprocess.run(
                [self.command, str(image_path), "--stdout"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Surya returned non-JSON output.") from exc
        text_rows: List[str] = []
        boxes: List[Dict[str, Any]] = []
        confidences: List[float] = []
        tables: List[Dict[str, Any]] = []
        for block in self._blocks(payload):
            html = str(block.get("html") or "")
            text = str(block.get("text") or _html_text(html)).strip()
            if not text:
                continue
            confidence = block.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            box = _box_from_polygon(block.get("bbox") or block.get("polygon"))
            text_rows.append(text)
            boxes.append(
                {
                    "text": text,
                    **(box or {}),
                    "confidence": confidence,
                    "coordinate_unit": "pixel",
                    "label": str(block.get("label") or ""),
                }
            )
            if confidence is not None:
                confidences.append(confidence)
            if "<table" in html.casefold():
                tables.append({"html": html, "bbox": block.get("bbox")})
        return OCRResult(
            text="\n".join(text_rows),
            bounding_boxes=boxes,
            confidence=(sum(confidences) / len(confidences) if confidences else None),
            tables=tables,
            metadata={"engine": "surya", "language": language},
        )


class FallbackOCRProvider:
    """Try configured providers in order and expose the engine that succeeded."""

    def __init__(self, providers: List[OCRProvider]) -> None:
        self.providers = providers

    def is_available(self) -> bool:
        return any(provider.is_available() for provider in self.providers)

    def extract(self, image_bytes: bytes, language: str) -> OCRResult:
        errors: List[str] = []
        for provider in self.providers:
            if not provider.is_available():
                errors.append(f"{provider.__class__.__name__}: unavailable")
                continue
            try:
                return provider.extract(image_bytes, language)
            except Exception as exc:
                errors.append(f"{provider.__class__.__name__}: {exc}")
                logger.warning("OCR provider failed; trying fallback: %s", errors[-1])
        raise RuntimeError("No OCR provider succeeded: " + "; ".join(errors))


def create_ocr_provider(engine: Optional[str] = None) -> OCRProvider:
    """Create the configured OCR provider without loading heavy models eagerly."""
    selected = (engine or settings.OCR_ENGINE or "paddle").strip().lower()
    factories = {
        "paddle": PaddleOCRProvider,
        "paddleocr": PaddleOCRProvider,
        "surya": SuryaOCRProvider,
        "tesseract": TesseractOCRProvider,
    }
    if selected == "auto":
        primary: OCRProvider = FallbackOCRProvider(
            [PaddleOCRProvider(), SuryaOCRProvider(), TesseractOCRProvider()]
        )
        return primary
    if selected not in factories:
        raise ValueError(
            f"Unsupported OCR_ENGINE {selected!r}; use paddle, surya, tesseract, or auto."
        )
    primary = factories[selected]()
    fallback_name = settings.OCR_FALLBACK_ENGINE
    if fallback_name and fallback_name != selected and fallback_name in factories:
        return FallbackOCRProvider([primary, factories[fallback_name]()])
    return primary
