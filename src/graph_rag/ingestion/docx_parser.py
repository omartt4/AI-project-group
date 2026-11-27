"""
DOCX Parser module.

Extracts text from Microsoft Word (.docx) documents using python-docx.
Supports both file paths and in-memory bytes streams.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from docx import Document

from graph_rag.config import settings
from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.ocr import OCRProvider, create_ocr_provider
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument


class DocxParser:
    """Extracts text and headings from DOCX files."""

    def __init__(
        self,
        ocr_provider: Optional[OCRProvider] = None,
        ocr_languages: Optional[str] = None,
    ) -> None:
        self.ocr_provider = ocr_provider or create_ocr_provider()
        self.ocr_languages = ocr_languages or settings.OCR_LANGUAGES

    def parse(self, source: Union[str, Path, bytes], filename: str = "document.docx") -> ParsedDocument:
        """Parse a .docx document from path or bytes."""
        if isinstance(source, bytes):
            file_stream = io.BytesIO(source)
            doc = Document(file_stream)
        else:
            path = Path(source)
            filename = path.name
            doc = Document(str(path))

        paragraphs: List[str] = []
        tables: List[Dict[str, Any]] = []
        ocr_texts: List[str] = []
        bounding_boxes: List[Dict[str, Any]] = []
        extracted_title = ""

        # Extract paragraphs and detect heading as title
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs.append(text)
                if not extracted_title and (p.style.name.startswith("Heading") or p.style.name == "Title"):
                    extracted_title = text

        for table_index, table in enumerate(doc.tables, start=1):
            raw_rows = [
                [cell.text.strip() for cell in row.cells]
                for row in table.rows
            ]
            if not raw_rows:
                continue
            width = max(len(row) for row in raw_rows)
            columns = self._unique_columns(
                [
                    raw_rows[0][index] if index < len(raw_rows[0]) else ""
                    for index in range(width)
                ]
            )
            rows = [
                {
                    columns[index]: (
                        row[index] if index < len(row) else ""
                    )
                    for index in range(width)
                }
                for row in raw_rows[1:]
            ]
            table_name = f"table_{table_index}"
            tables.append(
                {"table_name": table_name, "columns": columns, "rows": rows}
            )
            paragraphs.append(
                f"Table: {table_name}\nColumns: {' | '.join(columns)}\n"
                + "\n".join(
                    json.dumps(row, ensure_ascii=False) for row in rows
                )
            )

        for relationship in doc.part.rels.values():
            if not relationship.reltype.endswith("/image"):
                continue
            result = self.ocr_provider.extract(
                relationship.target_part.blob,
                language=self.ocr_languages,
            )
            if result.text.strip():
                ocr_texts.append(result.text.strip())
                paragraphs.append(
                    f"Image OCR ({relationship.rId}):\n{result.text.strip()}"
                )
            bounding_boxes.extend(
                {**box, "image_id": relationship.rId}
                for box in result.bounding_boxes
            )

        full_text = "\n\n".join(paragraphs)

        if not extracted_title and paragraphs:
            extracted_title = paragraphs[0][:120]
        if not extracted_title:
            extracted_title = Path(filename).stem.replace("_", " ")

        ocr_text = "\n\n".join(ocr_texts)
        page = PageContent(
            page_number=1,
            text=full_text,
            table_data=tables,
            ocr_text=ocr_text,
            source_offsets=[{"start": 0, "end": len(full_text)}] if full_text else [],
            bounding_boxes=bounding_boxes,
            extraction_method=(
                "embedded_text+tesseract_ocr" if ocr_text else "embedded_text"
            ),
            language=detect_language(full_text),
        )

        return ParsedDocument(
            filename=filename,
            title=extracted_title,
            total_pages=1,
            full_text=full_text,
            pages=[page] if full_text else [],
            metadata={"format": "docx"},
            table_data=tables,
            language=page.language,
        )

    @staticmethod
    def _unique_columns(columns: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for index, column in enumerate(columns, start=1):
            name = column or f"column_{index}"
            count = seen.get(name, 0) + 1
            seen[name] = count
            unique.append(name if count == 1 else f"{name}_{count}")
        return unique


docx_parser = DocxParser()
