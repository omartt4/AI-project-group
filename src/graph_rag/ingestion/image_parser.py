"""OCR parser for standalone image documents."""

from pathlib import Path
from typing import Union

from graph_rag.config import settings
from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.ocr import OCRProvider, create_ocr_provider
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument


class ImageParser:
    """Extract text and OCR provenance from a standalone image."""

    def __init__(
        self,
        ocr_provider: OCRProvider | None = None,
        ocr_languages: str | None = None,
    ) -> None:
        self.ocr_provider = ocr_provider or create_ocr_provider()
        self.ocr_languages = ocr_languages or settings.OCR_LANGUAGES

    def parse(
        self,
        source: Union[str, Path, bytes],
        filename: str = "image.png",
    ) -> ParsedDocument:
        image_bytes = (
            source if isinstance(source, bytes) else Path(source).read_bytes()
        )
        result = self.ocr_provider.extract(
            image_bytes,
            language=self.ocr_languages,
        )
        text = result.text.strip()
        page = PageContent(
            page_number=1,
            text=text,
            ocr_text=text,
            source_offsets=[{"start": 0, "end": len(text)}] if text else [],
            bounding_boxes=result.bounding_boxes,
            extraction_method="tesseract_ocr",
            confidence=result.confidence,
            language=detect_language(text),
        )
        return ParsedDocument(
            filename=filename,
            title=Path(filename).stem.replace("_", " "),
            total_pages=1,
            full_text=text,
            pages=[page],
            metadata={"format": Path(filename).suffix.lower().lstrip(".")},
            language=page.language,
        )


image_parser = ImageParser()
