"""
File Handler module.

Routes supported document formats to their format-specific parsers.
"""

import hashlib
import mimetypes
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from graph_rag.ingestion.docx_parser import docx_parser
from graph_rag.ingestion.image_parser import image_parser
from graph_rag.ingestion.language import detect_language
from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument, pdf_parser
from graph_rag.ingestion.pptx_parser import pptx_parser
from graph_rag.ingestion.structured_parser import structured_file_parser


class FileHandler:
    """Routes supported extensions to parsers and permits new parser plugins."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Callable[..., ParsedDocument]] = {
            ".pdf": lambda source, filename: pdf_parser.parse(source, filename=filename),
            ".docx": lambda source, filename: docx_parser.parse(source, filename=filename),
            ".pptx": lambda source, filename: pptx_parser.parse(source, filename=filename),
            ".bmp": lambda source, filename: image_parser.parse(source, filename=filename),
            ".jpeg": lambda source, filename: image_parser.parse(source, filename=filename),
            ".jpg": lambda source, filename: image_parser.parse(source, filename=filename),
            ".png": lambda source, filename: image_parser.parse(source, filename=filename),
            ".tif": lambda source, filename: image_parser.parse(source, filename=filename),
            ".tiff": lambda source, filename: image_parser.parse(source, filename=filename),
            ".webp": lambda source, filename: image_parser.parse(source, filename=filename),
            ".csv": structured_file_parser.parse,
            ".xlsx": structured_file_parser.parse,
            ".json": structured_file_parser.parse,
            ".parquet": structured_file_parser.parse,
        }

    def register_parser(
        self,
        extensions: Union[str, tuple[str, ...]],
        parser: Callable[..., ParsedDocument],
    ) -> None:
        """Register a parser for one or more filename extensions."""
        for extension in (extensions,) if isinstance(extensions, str) else extensions:
            normalized = extension.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            self._parsers[normalized] = parser

    def is_supported(self, filename: str) -> bool:
        """Check if file extension is supported for ingestion."""
        ext = Path(filename).suffix.lower()
        return ext in self._parsers or ext in {".txt", ".md"}

    def parse(
        self,
        source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> ParsedDocument:
        """
        Parse document content into standardized ParsedDocument.
        Accepts either file path or raw bytes.
        """
        if not filename:
            if isinstance(source, (str, Path)):
                filename = Path(source).name
            else:
                filename = "uploaded_file.bin"

        ext = Path(filename).suffix.lower()

        parser = self._parsers.get(ext)
        if parser:
            parsed = parser(source, filename=filename)
        elif ext in {".txt", ".md"}:
            parsed = self._parse_text(source, filename=filename)
        else:
            supported = ", ".join(sorted(self._parsers.keys() | {".txt", ".md"}))
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported formats: {supported}"
            )

        if not content_hash:
            hasher = hashlib.sha256()
            if isinstance(source, bytes):
                hasher.update(source)
            else:
                with open(source, "rb") as file:
                    for block in iter(lambda: file.read(65536), b""):
                        hasher.update(block)
            content_hash = hasher.hexdigest()
        parsed.document_id = content_hash
        parsed.content_hash = content_hash
        parsed.file_type = ext.lstrip(".")
        parsed.mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if parsed.language == "und":
            parsed.language = detect_language(parsed.full_text)
        for page in parsed.pages:
            if page.language == "und":
                page.language = detect_language(page.text)
            if page.text and not page.source_offsets:
                page.source_offsets = [{"start": 0, "end": len(page.text)}]
        return parsed

    def _parse_text(self, source: Union[str, Path, bytes], filename: str) -> ParsedDocument:
        """Parse plain text or markdown file."""
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace").strip()
        else:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        title = lines[0][:120] if lines else Path(filename).stem.replace("_", " ")

        return ParsedDocument(
            filename=filename,
            title=title,
            total_pages=1,
            full_text=text,
            pages=[PageContent(page_number=1, text=text)] if text else [],
            metadata={"format": Path(filename).suffix.lower().lstrip(".")},
        )


file_handler = FileHandler()
