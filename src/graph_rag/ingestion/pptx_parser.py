"""
PPTX Parser module.

Extracts text from Microsoft PowerPoint (.pptx) presentations using python-pptx.
Supports both file paths and in-memory byte streams.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from graph_rag.config import settings
from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.ocr import OCRProvider, TesseractOCRProvider
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument


class PptxParser:
    """Extracts text from PPTX slides."""

    def __init__(
        self,
        ocr_provider: Optional[OCRProvider] = None,
        ocr_languages: Optional[str] = None,
    ) -> None:
        self.ocr_provider = ocr_provider or TesseractOCRProvider()
        self.ocr_languages = ocr_languages or settings.OCR_LANGUAGES

    def parse(self, source: Union[str, Path, bytes], filename: str = "presentation.pptx") -> ParsedDocument:
        """Parse a .pptx presentation from path or bytes."""
        if isinstance(source, bytes):
            file_stream = io.BytesIO(source)
            prs = Presentation(file_stream)
        else:
            path = Path(source)
            filename = path.name
            prs = Presentation(str(path))

        pages: List[PageContent] = []
        full_text_parts: List[str] = []
        document_tables: List[Dict[str, Any]] = []
        extracted_title = ""

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_lines: List[str] = []
            slide_tables: List[Dict[str, Any]] = []
            slide_ocr_texts: List[str] = []
            slide_boxes: List[Dict[str, Any]] = []
            image_index = 0
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_lines.append(text)

                if shape.has_table:
                    table = shape.table
                    raw_rows = [
                        [cell.text.strip() for cell in row.cells]
                        for row in table.rows
                    ]
                    if raw_rows:
                        width = max(len(row) for row in raw_rows)
                        columns = self._unique_columns(
                            [
                                raw_rows[0][index]
                                if index < len(raw_rows[0])
                                else ""
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
                        table_name = f"slide_{slide_idx}_table_{len(slide_tables) + 1}"
                        table_data = {
                            "table_name": table_name,
                            "columns": columns,
                            "rows": rows,
                        }
                        slide_tables.append(table_data)
                        slide_lines.append(
                            f"Table: {table_name}\nColumns: {' | '.join(columns)}\n"
                            + "\n".join(
                                json.dumps(row, ensure_ascii=False)
                                for row in rows
                            )
                        )

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_index += 1
                    result = self.ocr_provider.extract(
                        shape.image.blob,
                        language=self.ocr_languages,
                    )
                    if result.text.strip():
                        slide_ocr_texts.append(result.text.strip())
                        slide_lines.append(
                            f"Image OCR ({image_index}):\n{result.text.strip()}"
                        )
                    slide_boxes.extend(
                        {
                            **box,
                            "image_index": image_index,
                            "slide_x": int(shape.left),
                            "slide_y": int(shape.top),
                            "slide_width": int(shape.width),
                            "slide_height": int(shape.height),
                        }
                        for box in result.bounding_boxes
                    )

            # Slide notes (if present)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    slide_lines.append(f"[Notes: {notes_text}]")

            slide_content = "\n".join(slide_lines)
            ocr_text = "\n\n".join(slide_ocr_texts)
            page = PageContent(
                page_number=slide_idx,
                text=slide_content,
                table_data=slide_tables,
                ocr_text=ocr_text,
                source_offsets=(
                    [{"start": 0, "end": len(slide_content)}]
                    if slide_content
                    else []
                ),
                bounding_boxes=slide_boxes,
                extraction_method=(
                    "embedded_text+tesseract_ocr" if ocr_text else "embedded_text"
                ),
                language=detect_language(slide_content),
            )
            pages.append(page)
            document_tables.extend(slide_tables)
            if slide_content:
                full_text_parts.append(slide_content)
                if not extracted_title:
                    extracted_title = slide_lines[0]

        if not extracted_title:
            extracted_title = Path(filename).stem.replace("_", " ")

        return ParsedDocument(
            filename=filename,
            title=extracted_title,
            total_pages=len(prs.slides),
            full_text="\n\n".join(full_text_parts),
            pages=pages,
            metadata={"format": "pptx"},
            table_data=document_tables,
            language=detect_language("\n\n".join(full_text_parts)),
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


pptx_parser = PptxParser()
