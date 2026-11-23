"""
PDF Parser module.

Extracts text, page boundaries, and document metadata from PDF files using PyMuPDF (fitz).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import pymupdf as fitz  # PyMuPDF

from graph_rag.config import settings
from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.ocr import OCRProvider, TesseractOCRProvider


@dataclass
class PageContent:
    """Extracted text and metadata for a single page."""
    page_number: int
    text: str
    sheet_name: str = ""
    table_name: str = ""
    section: str = ""
    table_data: List[Dict[str, Any]] = field(default_factory=list)
    ocr_text: str = ""
    source_offsets: List[Dict[str, int]] = field(default_factory=list)
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    extraction_method: str = "embedded_text"
    confidence: Optional[float] = None
    language: str = "und"


@dataclass
class ParsedDocument:
    """Standardized representation of a parsed document."""
    filename: str
    title: str
    total_pages: int
    full_text: str
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    table_data: List[Dict[str, Any]] = field(default_factory=list)
    document_id: str = ""
    content_hash: str = ""
    file_type: str = ""
    mime_type: str = ""
    language: str = "und"
    document_type: str = "unknown"


class PDFParser:
    """Extracts text and metadata from PDF files."""

    def __init__(
        self,
        ocr_provider: Optional[OCRProvider] = None,
        ocr_dpi: int = 200,
        ocr_languages: Optional[str] = None,
    ) -> None:
        if ocr_dpi < 72:
            raise ValueError("OCR_DPI must be at least 72")
        self.ocr_provider = ocr_provider or TesseractOCRProvider()
        self.ocr_dpi = ocr_dpi
        self.ocr_languages = ocr_languages or settings.OCR_LANGUAGES

    def parse(self, source: Union[str, Path, bytes], filename: str = "document.pdf") -> ParsedDocument:
        """Parse a PDF from a file path or in-memory bytes."""
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            path = Path(source)
            filename = path.name
            doc = fitz.open(str(path))

        try:
            pages: List[PageContent] = []
            full_text_parts: List[str] = []
            document_tables: List[Dict[str, Any]] = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                page_tables, table_texts = self._extract_tables(page, page_idx + 1)
                document_tables.extend(page_tables)
                for table, table_text in zip(page_tables, table_texts):
                    cell_values = [
                        value
                        for row in table["rows"]
                        for value in row.values()
                        if value
                    ]
                    if not cell_values or any(
                        value not in page_text for value in cell_values
                    ):
                        page_text = "\n\n".join(
                            part for part in (page_text, table_text) if part
                        )

                ocr_text_parts: List[str] = []
                bounding_boxes: List[Dict[str, Any]] = []
                confidences: List[float] = []
                extraction_methods = []
                if page_text:
                    extraction_methods.append("embedded_text")

                has_usable_text = len(re.findall(r"\w", page_text, flags=re.UNICODE)) >= 30
                page_images = page.get_images(full=True)
                if not has_usable_text and page_images:
                    pixmap = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
                    result = self.ocr_provider.extract(
                        pixmap.tobytes("png"),
                        language=self.ocr_languages,
                    )
                    ocr_text = result.text.strip()
                    if ocr_text:
                        ocr_text_parts.append(ocr_text)
                        extraction_methods.append("tesseract_ocr")
                        page_text = "\n\n".join(
                            part for part in (page_text, ocr_text) if part
                        )
                    if result.confidence is not None:
                        confidences.append(result.confidence)
                    scale = 72 / self.ocr_dpi
                    bounding_boxes.extend(
                        self._scale_boxes(result.bounding_boxes, scale, scale)
                    )
                elif has_usable_text and page_images:
                    for xref in dict.fromkeys(image[0] for image in page_images):
                        image = doc.extract_image(xref)
                        image_bytes = image.get("image")
                        if not image_bytes:
                            continue
                        result = self.ocr_provider.extract(
                            image_bytes,
                            language=self.ocr_languages,
                        )
                        image_text = result.text.strip()
                        if image_text:
                            ocr_text_parts.append(image_text)
                            page_text = "\n\n".join(
                                (page_text, f"Image OCR ({xref}):\n{image_text}")
                            )
                            extraction_methods.append("tesseract_ocr")
                        if result.confidence is not None:
                            confidences.append(result.confidence)
                        image_width = max(int(image.get("width", 0)), 1)
                        image_height = max(int(image.get("height", 0)), 1)
                        for rect in page.get_image_rects(xref):
                            bounding_boxes.extend(
                                self._scale_boxes(
                                    result.bounding_boxes,
                                    rect.width / image_width,
                                    rect.height / image_height,
                                    offset_x=rect.x0,
                                    offset_y=rect.y0,
                                    image_xref=xref,
                                )
                            )

                ocr_text = "\n\n".join(ocr_text_parts)
                page_content = PageContent(
                    page_number=page_idx + 1,
                    text=page_text,
                    table_data=page_tables,
                    ocr_text=ocr_text,
                    bounding_boxes=bounding_boxes,
                    extraction_method="+".join(extraction_methods) or "empty",
                    confidence=(
                        sum(confidences) / len(confidences)
                        if confidences
                        else None
                    ),
                    language=detect_language(page_text),
                )
                page_content.source_offsets = (
                    [{"start": 0, "end": len(page_text)}] if page_text else []
                )
                pages.append(page_content)
                if page_text:
                    full_text_parts.append(page_text)

            meta = doc.metadata or {}
            extracted_title = meta.get("title", "").strip()

            # If embedded title is absent or generic, attempt to extract from top lines of page 1
            if not extracted_title or extracted_title.lower().endswith(".pdf"):
                if pages:
                    first_lines = [
                        line.strip()
                        for line in pages[0].text.split("\n")
                        if len(line.strip()) > 5
                    ]
                    if first_lines:
                        extracted_title = first_lines[0]

            if not extracted_title:
                extracted_title = Path(filename).stem.replace("_", " ")

            return ParsedDocument(
                filename=filename,
                title=extracted_title,
                total_pages=len(doc),
                full_text="\n\n".join(full_text_parts),
                pages=pages,
                metadata={
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "keywords": meta.get("keywords", ""),
                    "creator": meta.get("creator", ""),
                },
                table_data=document_tables,
            )
        finally:
            doc.close()

    @staticmethod
    def _extract_tables(
        page: fitz.Page,
        page_number: int,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Extract PDF tables into row records and a searchable text form."""
        finder = page.find_tables()
        tables: List[Dict[str, Any]] = []
        table_texts: List[str] = []
        for table_index, table in enumerate(finder.tables, start=1):
            raw_rows = table.extract()
            if not raw_rows:
                continue
            width = max((len(row) for row in raw_rows), default=0)
            if width == 0:
                continue
            columns = [
                str(raw_rows[0][index] or "").strip() or f"column_{index + 1}"
                for index in range(width)
            ]
            columns = PDFParser._unique_columns(columns)
            records = [
                {
                    columns[index]: (
                        str(row[index]).strip()
                        if index < len(row) and row[index] is not None
                        else ""
                    )
                    for index in range(width)
                }
                for row in raw_rows[1:]
            ]
            table_name = f"page_{page_number}_table_{table_index}"
            table_data = {
                "table_name": table_name,
                "columns": columns,
                "rows": records,
            }
            tables.append(table_data)
            table_texts.append(
                f"Table: {table_name}\nColumns: {' | '.join(columns)}\n"
                + "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in records
                )
            )
        return tables, table_texts

    @staticmethod
    def _unique_columns(columns: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for index, column in enumerate(columns, start=1):
            count = seen.get(column, 0) + 1
            seen[column] = count
            unique.append(column if count == 1 else f"{column}_{count}")
            if not column:
                unique[-1] = f"column_{index}"
        return unique

    @staticmethod
    def _scale_boxes(
        boxes: List[Dict[str, Any]],
        scale_x: float,
        scale_y: float,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        image_xref: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                **box,
                "x": offset_x + box["x"] * scale_x,
                "y": offset_y + box["y"] * scale_y,
                "width": box["width"] * scale_x,
                "height": box["height"] * scale_y,
                "coordinate_unit": "point",
                **({"image_xref": image_xref} if image_xref is not None else {}),
            }
            for box in boxes
        ]


# Global parser instance
pdf_parser = PDFParser()
