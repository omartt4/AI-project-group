"""
Phase 0: Build corpus manifest from PDFs
Phase 1: Extract text and chunk into overlapping pieces
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

from lab_common import (
    get_config,
    parse_field_from_filename,
    print_section,
    print_warning,
    print_error
)


def verify_all_models():
    """Import and run model verification."""
    from lab_models import main as verify_models_main
    # Temporarily replace sys.argv to run verification
    old_argv = sys.argv
    sys.argv = ["lab_models.py"]
    try:
        # Capture exit code
        try:
            verify_models_main()
        except SystemExit as e:
            if e.code != 0:
                sys.exit(e.code)
    finally:
        sys.argv = old_argv


class PDFParser:
    """Parse PDFs using configured library."""
    
    def __init__(self, library: str = "pymupdf"):
        self.library = library
        if library == "pymupdf":
            try:
                import fitz
                self.fitz = fitz
            except ImportError:
                print_warning("PyMuPDF not available, falling back to pypdf")
                self.library = "pypdf"
        
        if self.library == "pypdf":
            try:
                import pypdf
                self.pypdf = pypdf
            except ImportError:
                print_error("Neither PyMuPDF nor pypdf is available")
                raise
    
    def extract_pages(self, pdf_path: Path) -> List[Dict]:
        """Extract text from each page with metadata."""
        pages = []
        
        if self.library == "pymupdf":
            doc = self.fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
            doc.close()
        
        elif self.library == "pypdf":
            reader = self.pypdf.PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
        
        return pages


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Join hyphenated line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Remove page numbers like "Page 1", "1", "Page 1 of 10", etc.
    text = re.sub(r'\n\s*[Pp]age\s*\d+(\s+of\s+\d+)?\s*\n', '\n', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove common header/footer patterns
    text = re.sub(r'\n\s*\d+\s*-\s*\d+\s*\n', '\n', text)  # "1 - 10"
    text = re.sub(r'\n\s*[A-Z][a-z]+\s+\d{4}\s*\n', '\n', text)  # "Conference 2024"
    
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()


def detect_and_remove_headers_footers(pages: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    Detect and remove headers/footers that appear on >threshold% of pages.
    Returns cleaned pages.
    """
    if len(pages) < 3:
        return pages
    
    # Collect first and last lines from each page
    first_lines = []
    last_lines = []
    
    for page in pages:
        lines = page["text"].split("\n")
        if lines:
            first_lines.append(lines[0].strip())
            last_lines.append(lines[-1].strip())
    
    # Find lines that appear on >threshold% of pages
    def find_repeated(lines):
        counter = Counter(lines)
        threshold_count = int(len(lines) * threshold)
        return {line for line, count in counter.items() if count >= threshold_count}
    
    repeated_first = find_repeated(first_lines)
    repeated_last = find_repeated(last_lines)
    
    # Remove repeated headers/footers
    cleaned = []
    for page in pages:
        lines = page["text"].split("\n")
        if lines:
            if lines[0].strip() in repeated_first:
                lines = lines[1:]
            if lines and lines[-1].strip() in repeated_last:
                lines = lines[:-1]
        page["text"] = "\n".join(lines)
        cleaned.append(page)
    
    return cleaned


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate
        return len(text.split())


class RecursiveChunker:
    """Recursive text chunker that respects word boundaries."""
    
    def __init__(self, size: int, overlap: int, unit: str = "tokens", 
                 separators: Optional[List[str]] = None):
        self.size = size
        self.overlap = overlap
        self.unit = unit
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str) -> List[Dict]:
        """Split text into chunks."""
        if self.unit == "tokens":
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_chars(text)
    
    def _chunk_by_tokens(self, text: str) -> List[Dict]:
        """Chunk by token count."""
        chunks = []
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = min(start + self.size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "n_tokens": len(chunk_tokens)
            })
            
            start += (self.size - self.overlap)
            chunk_index += 1
        
        return chunks
    
    def _chunk_by_chars(self, text: str) -> List[Dict]:
        """Chunk by character count with recursive splitting."""
        chunks = []
        
        def split_recursive(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]
            
            sep = separators[0]
            if sep == "":
                # Word-level splitting as last resort
                words = text.split()
                result = []
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= self.size:
                        current += (" " if current else "") + word
                    else:
                        if current:
                            result.append(current)
                        current = word
                if current:
                    result.append(current)
                return result
            
            parts = text.split(sep)
            result = []
            current = ""
            
            for part in parts:
                if len(current) + len(part) + len(sep) <= self.size:
                    current += (sep if current else "") + part
                else:
                    if current:
                        # Try to split current further
                        sub_parts = split_recursive(current, separators[1:])
                        result.extend(sub_parts)
                    current = part
            
            if current:
                sub_parts = split_recursive(current, separators[1:])
                result.extend(sub_parts)
            
            return result
        
        raw_chunks = split_recursive(text, self.separators)
        
        # Add overlap
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "chunk_index": i,
                "text": chunk_text,
                "n_tokens": count_tokens(chunk_text)
            })
        
        return chunks


def build_manifest(pdf_dir: str, config) -> List[Dict]:
    """
    Phase 0: Build manifest.csv from PDFs.
    Returns list of paper metadata.
    """
    print_section("Phase 0: Building Corpus Manifest")
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print_error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
    
    # Find all PDFs
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    if len(pdf_files) != 9:
        print_warning(f"Expected 9 PDFs, found {len(pdf_files)}")
        print("PDF files:")
        for f in pdf_files:
            print(f"  - {f.name}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)
    
    papers = []
    parser = PDFParser(config.pdf_library)
    
    for pdf_file in pdf_files:
        # Parse field from filename
        field = parse_field_from_filename(pdf_file.name)
        if field is None:
            print_error(f"Filename does not match pattern: {pdf_file.name}")
            sys.exit(1)
        
        # Extract pages
        pages = parser.extract_pages(pdf_file)
        
        if not pages or all(not p["text"].strip() for p in pages):
            print_warning(f"No extractable text in: {pdf_file.name} (possibly scanned)")
        
        # Get title from metadata or first line
        title = pdf_file.stem
        if pages and pages[0]["text"].strip():
            first_line = pages[0]["text"].split("\n")[0].strip()
            if first_line:
                title = first_line
        
        # Clean pages
        pages = detect_and_remove_headers_footers(pages)
        for page in pages:
            page["text"] = clean_text(page["text"])
        
        # Count characters
        n_chars = sum(len(p["text"]) for p in pages)
        
        paper_id = pdf_file.stem
        
        papers.append({
            "paper_id": paper_id,
            "field": field,
            "title": title,
            "filename": pdf_file.name,
            "pages": len(pages),
            "n_chars": n_chars
        })
        
        print(f"✓ {paper_id} ({field}): {len(pages)} pages, {n_chars:,} chars")
    
    # Save manifest
    manifest_path = Path(config.data_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "field", "title", "filename", "pages", "n_chars"])
        writer.writeheader()
        writer.writerows(papers)
    
    print(f"\nSaved manifest to {manifest_path}")
    return papers


def chunk_papers(papers: List[Dict], config, cut_references: bool = False) -> List[Dict]:
    """
    Phase 1: Chunk papers into overlapping pieces.
    Returns list of chunks with metadata.
    """
    print_section("Phase 1: Chunking Papers")
    
    parser = PDFParser(config.pdf_library)
    chunker = RecursiveChunker(
        size=config.chunk_size,
        overlap=config.chunk_overlap,
        unit=config.chunk_unit,
        separators=config.separators
    )
    
    all_chunks = []
    
    for paper in papers:
        pdf_path = Path(config.pdf_dir) / paper["filename"]
        pages = parser.extract_pages(pdf_path)
        pages = detect_and_remove_headers_footers(pages)
        
        # Clean and concatenate
        full_text = ""
        page_offsets = []  # Track where each page starts
        
        for page in pages:
            cleaned = clean_text(page["text"])
            page_offsets.append(len(full_text))
            full_text += cleaned + "\n\n"
        
        # Optionally cut references
        if cut_references:
            ref_match = re.search(r'\n\s*[Rr]eferences?\s*\n', full_text)
            if ref_match:
                full_text = full_text[:ref_match.start()]
                print(f"  Cut references from {paper['paper_id']}")
        
        # Chunk
        chunks = chunker.chunk(full_text)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk_text = chunk["text"]
            char_start = full_text.find(chunk_text)
            char_end = char_start + len(chunk_text)
            
            # Find page range
            page_start = 0
            page_end = len(pages)
            for i, offset in enumerate(page_offsets):
                if offset <= char_start:
                    page_start = i + 1
                if offset >= char_end:
                    page_end = i
                    break
            
            chunk.update({
                "chunk_id": f"{paper['paper_id']}_{chunk['chunk_index']}",
                "paper_id": paper["paper_id"],
                "field": paper["field"],
                "page_start": page_start,
                "page_end": page_end,
                "char_start": char_start,
                "char_end": char_end,
                "entities": []  # Will be filled by Phase 1b
            })
            
            all_chunks.append(chunk)
        
        print(f"  {paper['paper_id']}: {len(chunks)} chunks")
    
    # Save chunks
    chunks_file = Path(config.data_dir) / f"chunks_{config.chunk_size}_{config.chunk_overlap}.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"\nSaved {len(all_chunks)} chunks to {chunks_file}")
    
    # Print stats
    token_counts = [c["n_tokens"] for c in all_chunks]
    print(f"\nChunk statistics:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Tokens: min={min(token_counts)}, mean={sum(token_counts)/len(token_counts):.1f}, max={max(token_counts)}")
    print(f"  Chunks under 50 tokens: {sum(1 for t in token_counts if t < 50)} ({100*sum(1 for t in token_counts if t < 50)/len(token_counts):.1f}%)")
    
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Phase 0+1: Build corpus and chunk papers")
    parser.add_argument("--pdf-dir", default=None, help="PDF directory (default: from config)")
    parser.add_argument("--quick", action="store_true", help="Use only 2 papers per field (6 total)")
    parser.add_argument("--cut-references", action="store_true", help="Remove references section")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in tokens (default: from config)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in tokens (default: from config)")
    args = parser.parse_args()
    
    # Verify models first
    print_section("Verifying OpenRouter Models")
    verify_all_models()
    
    config = get_config()
    
    if args.pdf_dir:
        config.pdf_dir = args.pdf_dir
    
    # Override chunk size/overlap if specified
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    elif args.chunk_size == 256:
        config.chunk_overlap = config.chunk_overlap_256
    elif args.chunk_size == 1024:
        config.chunk_overlap = config.chunk_overlap_1024
    
    # Privacy warning
    print_warning("Free endpoints log prompts. Only research PDFs will be sent, never personal documents.")
    
    # Build manifest
    papers = build_manifest(config.pdf_dir, config)
    
    # Quick mode: filter to 2 papers per field
    if args.quick:
        print("\nQuick mode: using 2 papers per field")
        field_counts = {}
        quick_papers = []
        for paper in papers:
            field = paper["field"]
            if field_counts.get(field, 0) < 2:
                quick_papers.append(paper)
                field_counts[field] = field_counts.get(field, 0) + 1
        papers = quick_papers
        print(f"Selected {len(papers)} papers for quick mode\n")
    
    # Chunk
    chunks = chunk_papers(papers, config, cut_references=args.cut_references)
    
    print_section("Done")
    print(f"Processed {len(papers)} papers into {len(chunks)} chunks")


if __name__ == "__main__":
    main()
