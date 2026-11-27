"""Isolated PaddleOCR HTTP service used by the GraphRAG application."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from paddleocr import PaddleOCR
from PIL import Image


app = FastAPI(title="GraphRAG PaddleOCR Service", version="1.0")


def _box(polygon: Any) -> dict[str, int]:
    if len(polygon) == 4 and all(isinstance(value, (int, float)) for value in polygon):
        x1, y1, x2, y2 = [float(value) for value in polygon]
    else:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return {
        "x": round(x1),
        "y": round(y1),
        "width": max(0, round(x2 - x1)),
        "height": max(0, round(y2 - y1)),
    }


@lru_cache(maxsize=4)
def _pipeline(language: str) -> PaddleOCR:
    try:
        return PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(lang=language, use_angle_cls=False)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = value
    else:
        payload = getattr(value, "json", None)
        payload = payload() if callable(payload) else payload
        if not isinstance(payload, dict) and hasattr(value, "to_dict"):
            payload = value.to_dict()
    if not isinstance(payload, dict):
        return {}
    return payload.get("res", payload) if isinstance(payload.get("res"), dict) else payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(request: Request, language: str = "") -> dict[str, Any]:
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image body")
    selected_language = language or os.getenv("PADDLE_OCR_LANGUAGE", "ar")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = _pipeline(selected_language).predict(np.asarray(image))
        texts: list[str] = []
        boxes: list[dict[str, Any]] = []
        scores: list[float] = []
        tables: list[dict[str, Any]] = []
        for result in results:
            payload = _mapping(result)
            rec_texts = payload.get("rec_texts") or []
            rec_scores = payload.get("rec_scores") or []
            rec_boxes = payload.get("rec_polys") or payload.get("dt_polys") or []
            for index, value in enumerate(rec_texts):
                text = str(value).strip()
                if not text:
                    continue
                score = float(rec_scores[index]) if index < len(rec_scores) else None
                geometry = _box(rec_boxes[index]) if index < len(rec_boxes) else {}
                texts.append(text)
                boxes.append(
                    {
                        "text": text,
                        **geometry,
                        "confidence": score,
                        "coordinate_unit": "pixel",
                    }
                )
                if score is not None:
                    scores.append(score)
            table_rows = payload.get("table_res_list") or payload.get("tables") or []
            if isinstance(table_rows, list):
                tables.extend(
                    value if isinstance(value, dict) else {"value": str(value)}
                    for value in table_rows
                )
        return {
            "text": "\n".join(texts),
            "bounding_boxes": boxes,
            "confidence": sum(scores) / len(scores) if scores else None,
            "tables": tables,
            "metadata": {"engine": "paddleocr", "language": selected_language},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
