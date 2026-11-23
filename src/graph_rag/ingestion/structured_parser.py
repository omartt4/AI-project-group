"""Parsers for tabular and structured document formats."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument

_ROWS_PER_PAGE = 50


class StructuredFileParser:
    """Convert structured files to searchable text while retaining their rows."""

    def parse(
        self,
        source: Union[str, Path, bytes],
        filename: str,
    ) -> ParsedDocument:
        extension = Path(filename).suffix.lower()
        content = source if isinstance(source, bytes) else Path(source).read_bytes()

        if extension == ".csv":
            frames = [("data", self._read_csv(content))]
        elif extension == ".xlsx":
            frames = [
                (str(sheet_name), frame)
                for sheet_name, frame in pd.read_excel(
                    io.BytesIO(content), sheet_name=None
                ).items()
            ]
        elif extension == ".parquet":
            frames = [("data", pd.read_parquet(io.BytesIO(content)))]
        elif extension == ".json":
            payload = json.loads(content.decode("utf-8-sig"))
            frames = [("data", self._json_frame(payload))]
        else:
            raise ValueError(f"Unsupported structured file format: '{extension}'")

        pages: List[PageContent] = []
        tables: List[Dict[str, Any]] = []
        for table_name, frame in frames:
            rows = self._records(frame)
            columns = [str(column) for column in frame.columns]
            tables.append(
                {
                    "table_name": table_name,
                    "columns": columns,
                    "rows": rows,
                }
            )
            for offset in range(0, max(1, len(rows)), _ROWS_PER_PAGE):
                page_rows = rows[offset : offset + _ROWS_PER_PAGE]
                page_text = self._format_table(table_name, columns, page_rows)
                page_language_text = " ".join(
                    str(value)
                    for row in page_rows
                    for value in row.values()
                    if value is not None
                )
                pages.append(
                    PageContent(
                        page_number=len(pages) + 1,
                        text=page_text,
                        sheet_name=table_name if extension == ".xlsx" else "",
                        table_name=table_name,
                        table_data=[
                            {
                                "table_name": table_name,
                                "columns": columns,
                                "rows": page_rows,
                            }
                        ],
                        language=detect_language(page_language_text),
                    )
                )

        full_text = "\n\n".join(page.text for page in pages)
        language_text = " ".join(
            str(value)
            for table in tables
            for row in table["rows"]
            for value in row.values()
            if value is not None
        )
        return ParsedDocument(
            filename=filename,
            title=Path(filename).stem.replace("_", " "),
            total_pages=len(pages),
            full_text=full_text,
            pages=pages,
            metadata={"format": extension.lstrip(".")},
            table_data=tables,
            language=detect_language(language_text),
        )

    @staticmethod
    def _read_csv(content: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="cp1252")

    @staticmethod
    def _json_frame(payload: Any) -> pd.DataFrame:
        if isinstance(payload, list):
            return pd.json_normalize(payload) if payload else pd.DataFrame()
        if isinstance(payload, dict):
            return pd.json_normalize(payload)
        return pd.DataFrame({"value": [payload]})

    @staticmethod
    def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            record: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    record[str(key)] = None
                    continue
                try:
                    missing = pd.isna(value)
                except (TypeError, ValueError):
                    missing = False
                try:
                    is_missing = bool(missing)
                except (TypeError, ValueError):
                    is_missing = False
                if is_missing:
                    record[str(key)] = None
                elif hasattr(value, "item"):
                    scalar = value.item()
                    record[str(key)] = (
                        scalar.isoformat()
                        if hasattr(scalar, "isoformat")
                        else scalar
                    )
                elif hasattr(value, "isoformat"):
                    record[str(key)] = value.isoformat()
                else:
                    record[str(key)] = value
            cleaned.append(record)
        return cleaned

    @staticmethod
    def _format_table(
        table_name: str,
        columns: List[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        lines = [f"Table: {table_name}", "Columns: " + " | ".join(columns)]
        lines.extend(
            json.dumps(row, ensure_ascii=False, default=str)
            for row in rows
        )
        return "\n".join(lines)


structured_file_parser = StructuredFileParser()
