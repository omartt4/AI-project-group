"""
Text Chunker module.

Splits parsed documents into overlapping chunks with metadata preservation.
Maintains context window boundaries (~1000 tokens with ~150 token overlap).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_rag.config import settings
from graph_rag.ingestion.pdf_parser import ParsedDocument


@dataclass
class TextChunk:
    """A semantic chunk of text with document metadata."""
    chunk_id: str
    document_name: str
    document_title: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Splits document text into overlapping chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_document(self, doc: ParsedDocument) -> List[TextChunk]:
        """
        Chunk a parsed document page by page or full text.
        Prefers page-aware chunking when pages are available.
        """
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        if doc.pages:
            for page in doc.pages:
                page_chunks = self._chunk_text(
                    text=page.text,
                    doc=doc,
                    page_number=page.page_number,
                    start_idx=global_chunk_idx,
                    sheet_name=page.sheet_name,
                    table_name=page.table_name,
                    section=page.section,
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    language=page.language,
                )
                chunks.extend(page_chunks)
                global_chunk_idx += len(page_chunks)
        else:
            chunks = self._chunk_text(
                text=doc.full_text,
                doc=doc,
                page_number=1,
                start_idx=0,
            )

        return chunks

    def _chunk_text(
        self,
        text: str,
        doc: ParsedDocument,
        page_number: int,
        start_idx: int,
        sheet_name: str = "",
        table_name: str = "",
        section: str = "",
        extraction_method: str = "",
        confidence: Optional[float] = None,
        language: str = "und",
    ) -> List[TextChunk]:
        """Split text by sentences/paragraphs into target chunk size."""
        if not text.split():
            return []

        # Preserve the existing word-based size estimate across English and Arabic.
        target_words = int(self.chunk_size * 0.75)
        target_words = max(1, target_words)
        overlap_words = min(int(self.chunk_overlap * 0.75), target_words - 1)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_words,
            chunk_overlap=max(0, overlap_words),
            length_function=lambda value: len(value.split()),
        )
        split_documents = splitter.create_documents([text])

        chunks: List[TextChunk] = []
        current_idx = start_idx
        search_from = 0
        for split_document in split_documents:
            chunk_text = split_document.page_content.strip()
            if not chunk_text:
                continue
            start_index = text.find(chunk_text, search_from)
            if start_index < 0:
                start_index = text.find(chunk_text)
            if start_index >= 0:
                words = chunk_text.split()
                overlap_text = " ".join(words[-overlap_words:]) if overlap_words else ""
                search_from = start_index + max(
                    1,
                    len(chunk_text) - len(overlap_text),
                )
            chunk_id = f"{Path(doc.filename).stem}_p{page_number}_c{current_idx}"
            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update(
                {
                    "document_id": doc.document_id,
                    "content_hash": doc.content_hash,
                    "file_type": doc.file_type,
                    "mime_type": doc.mime_type,
                    "document_name": doc.filename,
                    "document_title": doc.title,
                    "page": page_number,
                    "chunk_index": current_idx,
                    "section": section,
                    "sheet_name": sheet_name,
                    "table_name": table_name,
                    "language": (
                        language if language != "und" else doc.language
                    ),
                    "extraction_method": extraction_method,
                }
            )
            if confidence is not None:
                chunk_metadata["confidence"] = confidence
            if start_index >= 0:
                chunk_metadata["start_index"] = start_index
                chunk_metadata["source_end"] = start_index + len(chunk_text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_name=doc.filename,
                    document_title=doc.title,
                    chunk_index=current_idx,
                    page_number=page_number,
                    text=chunk_text,
                    token_count=int(len(chunk_text.split()) * 1.3),
                    metadata=chunk_metadata,
                )
            )
            current_idx += 1

        return chunks


chunker = TextChunker()
