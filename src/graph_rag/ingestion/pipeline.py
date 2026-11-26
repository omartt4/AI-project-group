"""
Ingestion Pipeline module.

Orchestrates full document ingestion with two core modes:
- Mode A (Batch / Startup): Indexes pdfs/ folder into ChromaDB and Neo4j.
- Mode B (Dynamic Upload): User uploads documents via Streamlit UI.
  Strictly enforces:
  - In-memory duplicate verification before saving to disk.
  - If duplicate: REJECT (do not save to disk, do not run backend chunking/embeddings/graph).
  - If new: save to pdfs/, run ingestion, append to metadata.csv, log in SQLite.
"""

import logging
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

from graph_rag.config import settings
from graph_rag.database.db import (
    delete_document_evidence,
    delete_document_pages,
    delete_structured_tables,
    delete_unresolved_references,
    extraction_lock,
    finish_ingestion_run,
    get_incomplete_ingestion_run,
    get_db_connection,
    has_incomplete_ingestion_run,
    save_document_pages,
    save_extraction_diagnostics,
    save_graph_evidence,
    save_structured_tables,
    save_unresolved_references,
    start_ingestion_run,
    update_ingestion_run,
)
from graph_rag.embeddings.vector_store import vector_store
from graph_rag.graph.entity_extractor import (
    ExtractedEntity,
    ExtractedMention,
    ExtractedRelation,
    ExtractedTopic,
    GraphExtractionError,
    GraphExtractionResult,
    entity_extractor,
)
from graph_rag.graph.entity_resolution import DocumentEntityRegistry, EntityResolver
from graph_rag.graph.graph_builder import graph_builder
from graph_rag.graph.type_normalization import TypeNormalizer
from graph_rag.graph.topic_taxonomy import normalize_topic
from graph_rag.ingestion.archive_handler import archive_handler
from graph_rag.ingestion.chunker import chunker
from graph_rag.ingestion.dedup_checker import compute_hash, dedup_checker
from graph_rag.ingestion.file_handler import file_handler
from graph_rag.ingestion.document_structure import (
    prepare_document_input,
    split_document_input,
)
from graph_rag.ingestion.metadata_manager import metadata_manager
from graph_rag.ingestion.pdf_parser import PDFParser
from graph_rag.ingestion.spacy_candidates import spacy_candidate_extractor
from graph_rag.llm.model_client import embedding_model_client, model_request_context

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "2"
EVIDENCE_MATCH_VERSION = "arabic-evidence-v2"


def _usable_document_title(*titles: str, filename: str) -> str:
    """Prefer a parsed/source title over generic office-export metadata."""
    for title in titles:
        value = " ".join(str(title or "").split())
        if value and not PDFParser._is_generic_title(value):
            return value
    return Path(filename).stem.replace("_", " ")


@dataclass
class IngestionResult:
    """Detailed summary of an ingestion operation."""
    filename: str
    status: str  # 'completed', 'skipped_duplicate', 'failed'
    title: str = ""
    is_duplicate: bool = False
    match_reason: str = ""
    num_chunks: int = 0
    file_hash: str = ""
    error_message: str = ""


class IngestionPipeline:
    """Unified document ingestion pipeline."""

    def __init__(self) -> None:
        self.meta_mgr = metadata_manager
        self.dedup = dedup_checker
        self.vector_db = vector_store
        self.graph_db = graph_builder

    @staticmethod
    def _batch_input_files(folder: Path, metadata_file: Path) -> List[Path]:
        """List supported documents without treating the catalog CSV as input."""
        metadata_path = metadata_file.resolve()
        return sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file()
                and path.resolve() != metadata_path
                and file_handler.is_supported(path.name)
            ),
            key=lambda path: path.name.casefold(),
        )

    # ── Mode A: Batch Ingestion for pdfs/ folder ──
    def ingest_batch_folder(
        self,
        folder_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        limit: Optional[int] = None,
        only_filename: Optional[str] = None,
        rebuild_graph: bool = False,
        refresh_extraction: bool = False,
        refresh_vectors: bool = False,
    ) -> List[IngestionResult]:
        """
        Run initial batch ingestion over pdfs/ folder.
        Idempotent: Skips any paper already marked as ingested.
        """
        folder = folder_path or settings.PDF_DIR
        input_files = (
            self._batch_input_files(folder, settings.METADATA_FILE)
            if folder.exists()
            else []
        )
        if only_filename:
            requested = Path(only_filename).name.casefold()
            input_files = [path for path in input_files if path.name.casefold() == requested]
            if not input_files:
                raise FileNotFoundError(
                    f"Supported document not found in {folder}: {Path(only_filename).name}"
                )
        if limit and limit > 0:
            input_files = input_files[:limit]
        total_files = len(input_files)
        results: List[IngestionResult] = []

        # Load existing metadata records to check ingestion status
        records = self.meta_mgr.get_all_records()
        existing_records = {
            Path(r.get("local_pdf", "")).name.lower(): r
            for r in records
            if r.get("local_pdf")
        }
        existing_hashes = {
            str(r.get("file_hash", "")).strip().lower(): r
            for r in records
            if r.get("file_hash") and r.get("ingested_at")
        }

        # Ensure Neo4j constraints
        self.graph_db.init_constraints()

        for idx, input_file in enumerate(input_files, start=1):
            filename = input_file.name
            fname_lower = filename.lower()
            rec = existing_records.get(fname_lower, {})

            file_hash = compute_hash(input_file)

            # A renamed copy of an ingested document must not be processed again.
            matching_content = existing_hashes.get(file_hash)
            if (
                matching_content
                and Path(matching_content.get("local_pdf", "")).name.casefold()
                != filename.casefold()
            ):
                results.append(
                    IngestionResult(
                        filename=filename,
                        status="skipped_duplicate",
                        title=matching_content.get("title", filename),
                        is_duplicate=True,
                        match_reason="Identical file content is already indexed",
                        file_hash=file_hash,
                    )
                )
                if progress_callback:
                    progress_callback(idx, total_files, f"Skipped (duplicate content): {filename}")
                continue

            # Skip only when the file contents are unchanged.
            already_ingested = bool(
                rec.get("ingested_at")
                and str(rec.get("file_hash", "")).strip().lower() == file_hash
            )
            has_pending_run = has_incomplete_ingestion_run(filename, file_hash)
            if (
                already_ingested
                and not has_pending_run
                and not rebuild_graph
                and not refresh_extraction
                and not refresh_vectors
            ):
                results.append(
                    IngestionResult(
                        filename=filename,
                        status="skipped_duplicate",
                        title=rec.get("title", filename),
                        is_duplicate=True,
                        match_reason="Already indexed in knowledge base",
                        file_hash=file_hash,
                    )
                )
                if progress_callback:
                    progress_callback(idx, total_files, f"Skipped (already indexed): {filename}")
                continue

            if progress_callback:
                progress_callback(idx, total_files, f"Ingesting: {filename}")

            run_id: Optional[int] = None
            try:
                run_id = start_ingestion_run(filename, file_hash, "batch")
                # 1. Parse text
                parsed_doc = file_handler.parse(
                    input_file,
                    filename=filename,
                    content_hash=file_hash,
                )
                extracted_metadata = parsed_doc.metadata
                doc_title = _usable_document_title(
                    parsed_doc.title,
                    rec.get("title", ""),
                    filename=filename,
                )
                topic = rec.get("topic") or ""
                update_ingestion_run(run_id, "parsed")
                if progress_callback:
                    progress_callback(idx, total_files, f"Parsed: {filename}")

                # 3. Chunk text
                chunks = chunker.chunk_document(parsed_doc)
                if progress_callback:
                    progress_callback(idx, total_files, f"Created {len(chunks)} chunks: {filename}")

                document_id = parsed_doc.document_id or file_hash

                # 4. Extract once per document (or load the versioned extraction cache).
                graph_result = self._run_model_stage(
                    run_id,
                    None,
                    lambda: self._extract_document_graph(
                        parsed_doc,
                        doc_title,
                        document_id,
                        file_hash=file_hash,
                        refresh_extraction=refresh_extraction,
                        cache_only=rebuild_graph and not refresh_extraction,
                        document_name=filename,
                    ),
                )
                update_ingestion_run(run_id, "graph_extracted")
                parsed_doc.document_type = graph_result.document_type
                parsed_doc.metadata["document_type"] = graph_result.document_type

                # 5. A graph rebuild normally reuses existing vectors. If a
                # new/partially failed document has no vectors, repair them
                # automatically instead of leaving it invisible to retrieval.
                existing_vector_chunks = self.vector_db.get_document_chunk_count(filename)
                old_hash = str(rec.get("file_hash", "")).strip().lower()
                content_changed = existing_vector_chunks > 0 and (
                    not rec.get("ingested_at") or old_hash != file_hash
                )
                if self._needs_vector_replace(
                    refresh_extraction or refresh_vectors,
                    content_changed,
                    existing_vector_chunks,
                ):
                    self._run_model_stage(
                        run_id,
                        None,
                        lambda: self.vector_db.replace_document_chunks(
                            filename, chunks
                        ),
                    )
                    if progress_callback:
                        progress_callback(
                            idx, total_files, f"Replaced refreshed vector chunks: {filename}"
                        )
                elif self._needs_vector_write(rebuild_graph, existing_vector_chunks):
                    self._run_model_stage(
                        run_id,
                        None,
                        lambda: self.vector_db.add_chunks(chunks),
                    )
                    if progress_callback:
                        progress_callback(idx, total_files, f"Embedded and stored {len(chunks)} chunks: {filename}")
                elif progress_callback:
                    progress_callback(
                        idx,
                        total_files,
                        f"Reused {existing_vector_chunks} existing vector chunks: {filename}",
                    )
                update_ingestion_run(run_id, "vectors_ready")

                # 6. Populate Neo4j only after extraction succeeds, so a failed
                # model request cannot leave a paper marked as completed.
                if rebuild_graph:
                    self.graph_db.clear_document_evidence(
                        filename,
                        document_id=document_id,
                    )
                if not self.graph_db.add_document_node(
                    document_name=filename,
                    title=doc_title,
                    document_id=document_id,
                    document_type=parsed_doc.document_type,
                    metadata={
                        "topic": topic,
                        "source": rec.get("source", "initial"),
                        "topic_proposals": [
                            asdict(proposal) for proposal in graph_result.topics
                        ],
                        "entity_names": [entity.name for entity in graph_result.entities],
                        "diagnostics": graph_result.diagnostics,
                    },
                ):
                    raise RuntimeError("Neo4j document write failed")
                if not self.graph_db.add_entities_and_relations(
                    filename,
                    graph_result,
                    document_id=document_id,
                ):
                    raise RuntimeError("Neo4j entity write failed")
                self._run_model_stage(
                    run_id,
                    None,
                    lambda: self.vector_db.upsert_entity_vectors(
                        graph_result.entities,
                        document_id,
                    ),
                )
                update_ingestion_run(run_id, "graph_written")
                if progress_callback:
                    progress_callback(idx, total_files, f"Updated Neo4j graph: {filename}")

                save_structured_tables(
                    filename, file_hash, parsed_doc.table_data
                )
                save_document_pages(filename, file_hash, parsed_doc.pages)
                save_unresolved_references(
                    filename, file_hash, graph_result.mentions
                )
                save_graph_evidence(
                    document_id,
                    filename,
                    file_hash,
                    graph_result,
                )
                save_extraction_diagnostics(
                    document_id,
                    filename,
                    file_hash,
                    getattr(graph_result, "diagnostics", []),
                )

                # 7. Update metadata.csv. A fresh corpus may not have a row yet.
                if rec:
                    self.meta_mgr.mark_as_ingested(
                        local_pdf=filename,
                        file_hash=file_hash,
                        file_type=parsed_doc.file_type,
                        mime_type=parsed_doc.mime_type,
                        language=parsed_doc.language,
                    )
                else:
                    self.meta_mgr.append_document({
                        "topic": topic,
                        "title": doc_title,
                        "authors": extracted_metadata.get("author", ""),
                        "abstract": "",
                        "categories": topic,
                        "local_pdf": filename,
                        "file_type": parsed_doc.file_type,
                        "mime_type": parsed_doc.mime_type,
                        "language": parsed_doc.language,
                        "source": "initial",
                        "file_hash": file_hash,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    })
                update_ingestion_run(run_id, "local_metadata_saved")
                finish_ingestion_run(run_id)

                results.append(
                    IngestionResult(
                        filename=filename,
                        status="completed",
                        title=doc_title,
                        is_duplicate=False,
                        num_chunks=len(chunks),
                        file_hash=file_hash,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to ingest {filename}: {e}")
                if run_id is not None:
                    try:
                        finish_ingestion_run(run_id, error_message=str(e))
                    except Exception:
                        logger.exception(
                            "Could not persist failed ingestion run for %s",
                            filename,
                        )
                results.append(
                    IngestionResult(
                        filename=filename,
                        status="failed",
                        title=rec.get("title", filename),
                        error_message=str(e),
                        file_hash=file_hash,
                    )
                )
                error_text = str(e).casefold()
                if any(
                    marker in error_text
                    for marker in (
                        "403 forbidden",
                        "402 payment required",
                        "insufficient credits",
                        "credit limit",
                        "usage limit",
                    )
                ):
                    logger.error("Stopping batch after a non-recoverable API billing or permission error")
                    break

        return results

    @staticmethod
    def _needs_vector_write(rebuild_graph: bool, existing_chunk_count: int) -> bool:
        """Write vectors for normal ingestion or repair a missing rebuild entry."""
        return not rebuild_graph or existing_chunk_count == 0

    @staticmethod
    def _needs_vector_replace(
        refresh_extraction: bool,
        content_changed: bool,
        existing_chunk_count: int,
    ) -> bool:
        """Replace stale vectors when parsing/extraction was explicitly refreshed."""
        return existing_chunk_count > 0 and (
            refresh_extraction or content_changed
        )

    @staticmethod
    def _run_model_stage(
        run_id: int,
        user_id: Optional[int],
        operation: Callable[[], Any],
    ) -> Any:
        """Attach ingestion correlation IDs to nested chat and embedding requests."""
        with model_request_context(run_id=f"ingestion:{run_id}", user_id=user_id):
            return operation()

    # ── Mode B: Dynamic User Upload Ingestion ──
    def ingest_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: Optional[int] = None,
        source_archive: Optional[str] = None,
    ) -> IngestionResult:
        """
        Process a user-uploaded file.
        STRICT POLICY:
        - In-memory check against metadata.csv first.
        - If duplicate: DO NOT save to disk, DO NOT run backend processing.
        - If new: save to pdfs/, run pipeline, update metadata.csv & SQLite.
        """
        if not file_handler.is_supported(filename):
            return IngestionResult(
                filename=filename,
                status="failed",
                error_message=f"Unsupported file format: {Path(filename).suffix.lower()}",
            )

        # Step 1: Resume an interrupted upload before treating its hash as a duplicate.
        file_hash = compute_hash(file_bytes)
        recoverable_run = get_incomplete_ingestion_run(file_hash, "upload")
        if recoverable_run:
            recovered_path = settings.PDF_DIR / recoverable_run["document_name"]
            if recovered_path.exists() and compute_hash(recovered_path) != file_hash:
                recoverable_run = None

        if recoverable_run:
            filename = str(recoverable_run["document_name"])
        else:
            dedup_res = self.dedup.check_bytes(file_bytes=file_bytes, filename=filename)
            if dedup_res.is_duplicate:
                logger.info(f"Duplicate upload rejected: {filename} ({dedup_res.message})")

                try:
                    with get_db_connection() as conn:
                        conn.execute(
                            """
                            INSERT INTO uploaded_documents
                            (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, status, error_message)
                            VALUES (?, ?, ?, ?, ?, ?, 'duplicate_skipped', ?)
                            """,
                            (
                                user_id,
                                filename,
                                Path(filename).suffix.lower().lstrip("."),
                                len(file_bytes),
                                file_hash,
                                source_archive,
                                dedup_res.message,
                            ),
                        )
                except Exception:
                    logger.exception("Could not log upload rejection for %s", filename)

                return IngestionResult(
                    filename=filename,
                    status="skipped_duplicate",
                    title=dedup_res.matched_doc.get("title", filename) if dedup_res.matched_doc else filename,
                    is_duplicate=True,
                    match_reason=dedup_res.message,
                    file_hash=file_hash,
                )

        # Step 2: Save a new file, or reuse the interrupted run's filename.
        target_path = settings.PDF_DIR / filename
        existing_metadata = self.meta_mgr.find_by_filename(filename)
        metadata_hash = str((existing_metadata or {}).get("file_hash") or "").strip().lower()
        name_conflicts_with_other_content = bool(
            metadata_hash and metadata_hash != file_hash
        )
        if (
            target_path.exists()
            and (not recoverable_run or compute_hash(target_path) != file_hash)
        ) or name_conflicts_with_other_content:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{file_hash[:6]}{suffix}"
            target_path = settings.PDF_DIR / filename

        run_id: Optional[int] = None
        previous_metadata = self.meta_mgr.find_by_filename(filename)
        can_rollback_existing_stores = (
            not previous_metadata
            or str(previous_metadata.get("file_hash") or "").strip().lower() == file_hash
        )
        metadata_added = False
        graph_write_started = False
        document_id = file_hash
        try:
            run_id = start_ingestion_run(filename, file_hash, "upload")
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            # Step 3: Parse Document
            parsed_doc = file_handler.parse(
                target_path,
                filename=filename,
                content_hash=file_hash,
            )
            update_ingestion_run(run_id, "parsed")
            doc_title = _usable_document_title(
                parsed_doc.title,
                filename=filename,
            )

            # Step 4: Chunk Text
            chunks = chunker.chunk_document(parsed_doc)
            update_ingestion_run(run_id, "chunked")

            document_id = parsed_doc.document_id or file_hash

            # Step 5: Extract the compact document graph once and cache it.
            extracted_metadata = parsed_doc.metadata
            doc_title = _usable_document_title(
                parsed_doc.title,
                filename=filename,
            )
            topic = ""
            graph_result = self._run_model_stage(
                run_id,
                user_id,
                lambda: self._extract_document_graph(
                    parsed_doc,
                    doc_title,
                    document_id,
                    file_hash=file_hash,
                    document_name=filename,
                ),
            )
            update_ingestion_run(run_id, "graph_extracted")
            parsed_doc.document_type = graph_result.document_type
            parsed_doc.metadata["document_type"] = graph_result.document_type

            # Step 6: Embed & Upsert into ChromaDB.
            self._run_model_stage(
                run_id,
                user_id,
                lambda: self.vector_db.add_chunks(chunks),
            )
            update_ingestion_run(run_id, "vectors_ready")

            # Step 7: Build Knowledge Graph in Neo4j.
            graph_write_started = True
            if not self.graph_db.add_document_node(
                document_name=filename,
                title=doc_title,
                document_id=document_id,
                document_type=parsed_doc.document_type,
                metadata={
                    "topic": topic,
                    "source": "upload",
                    "topic_proposals": [
                        asdict(proposal) for proposal in graph_result.topics
                    ],
                    "entity_names": [entity.name for entity in graph_result.entities],
                    "diagnostics": graph_result.diagnostics,
                },
            ):
                raise RuntimeError("Neo4j document write failed")
            if not self.graph_db.add_entities_and_relations(
                filename,
                graph_result,
                document_id=document_id,
            ):
                raise RuntimeError("Neo4j entity write failed")
            self._run_model_stage(
                run_id,
                user_id,
                lambda: self.vector_db.upsert_entity_vectors(
                    graph_result.entities,
                    document_id,
                ),
            )
            update_ingestion_run(run_id, "graph_written")

            save_structured_tables(filename, file_hash, parsed_doc.table_data)
            save_document_pages(filename, file_hash, parsed_doc.pages)
            save_unresolved_references(
                filename, file_hash, graph_result.mentions
            )
            save_graph_evidence(
                document_id,
                filename,
                file_hash,
                graph_result,
            )
            save_extraction_diagnostics(
                document_id,
                filename,
                file_hash,
                getattr(graph_result, "diagnostics", []),
            )
            update_ingestion_run(run_id, "local_data_saved")

            # Step 8: Append to metadata.csv
            new_meta_row = {
                "topic": topic,
                "title": doc_title,
                "authors": str(extracted_metadata.get("author") or ""),
                "abstract": "",
                "categories": "",
                "local_pdf": filename,
                "file_type": parsed_doc.file_type,
                "mime_type": parsed_doc.mime_type,
                "language": parsed_doc.language,
                "source": "upload",
                "file_hash": file_hash,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            if previous_metadata:
                self.meta_mgr.mark_as_ingested(
                    local_pdf=filename,
                    file_hash=file_hash,
                    file_type=parsed_doc.file_type,
                    mime_type=parsed_doc.mime_type,
                    language=parsed_doc.language,
                )
            else:
                self.meta_mgr.append_document(new_meta_row)
                metadata_added = True
            update_ingestion_run(run_id, "metadata_saved")

            # Step 9: Log success in SQLite
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO uploaded_documents
                    (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, num_chunks, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'completed')
                    """,
                    (
                        user_id,
                        filename,
                        Path(filename).suffix.lower().lstrip("."),
                        len(file_bytes),
                        file_hash,
                        source_archive,
                        len(chunks),
                    ),
                )
            update_ingestion_run(run_id, "upload_record_saved")
            finish_ingestion_run(run_id)

            return IngestionResult(
                filename=filename,
                status="completed",
                title=doc_title,
                is_duplicate=False,
                num_chunks=len(chunks),
                file_hash=file_hash,
            )

        except Exception as e:
            logger.error(f"Failed to ingest uploaded file {filename}: {e}")
            if graph_write_started and can_rollback_existing_stores:
                try:
                    if not self.graph_db.delete_document(
                        filename,
                        document_id=document_id,
                    ):
                        logger.error(
                            "Graph cleanup failed for interrupted upload %s",
                            filename,
                        )
                except Exception:
                    logger.exception(
                        "Could not remove partial graph data for %s",
                        filename,
                    )
            if can_rollback_existing_stores:
                try:
                    self.vector_db.delete_document_chunks(filename)
                except Exception:
                    logger.exception("Could not remove partial vectors for %s", filename)
                try:
                    self.vector_db.delete_document_entity_vectors(document_id)
                except Exception:
                    logger.exception(
                        "Could not remove partial entity vectors for %s",
                        filename,
                    )
                try:
                    delete_structured_tables(filename)
                except Exception:
                    logger.exception("Could not remove structured rows for failed upload %s", filename)
                try:
                    delete_document_pages(filename)
                except Exception:
                    logger.exception("Could not remove page data for failed upload %s", filename)
                try:
                    delete_unresolved_references(filename)
                except Exception:
                    logger.exception(
                        "Could not remove unresolved mentions for failed upload %s",
                        filename,
                    )
                try:
                    delete_document_evidence(document_id)
                except Exception:
                    logger.exception(
                        "Could not remove graph evidence for failed upload %s",
                        filename,
                    )
            # Clean up saved file if ingestion aborted
            if target_path.exists() and can_rollback_existing_stores:
                try:
                    target_path.unlink()
                except Exception:
                    logger.exception("Could not remove temporary upload %s", filename)

            if metadata_added:
                try:
                    self.meta_mgr.remove_document(filename)
                except Exception:
                    logger.exception("Could not remove failed upload metadata for %s", filename)

            if run_id is not None:
                try:
                    finish_ingestion_run(run_id, error_message=str(e))
                except Exception:
                    logger.exception("Could not persist failed upload run for %s", filename)

            # Log failure in SQLite
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO uploaded_documents
                        (user_id, filename, file_type, file_size_bytes, file_hash, source_archive, status, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, 'failed', ?)
                        """,
                        (
                            user_id,
                            filename,
                            Path(filename).suffix.lower().lstrip("."),
                            len(file_bytes),
                            file_hash,
                            source_archive,
                            str(e),
                        ),
                    )
            except Exception:
                logger.exception("Could not log failed upload for %s", filename)

            return IngestionResult(
                filename=filename,
                status="failed",
                is_duplicate=False,
                error_message=str(e),
                file_hash=file_hash,
            )

    @staticmethod
    def _split_failed_extraction_segment(
        segment: Any,
    ) -> List[Any]:
        """Split one failed segment without changing successful segments."""
        rendered_length = len(segment.render())
        retry_max_characters = max(4000, rendered_length // 2)
        if retry_max_characters >= rendered_length:
            return [segment]
        try:
            return split_document_input(
                segment,
                retry_max_characters,
                len,
            )
        except ValueError:
            return [segment]

    def _extract_segments_with_retry(
        self,
        extraction_chain: Any,
        segments: List[Any],
        *,
        document_id: str,
        document_name: str,
        retry_depth: int = 0,
    ) -> tuple[List[GraphExtractionResult], List[Exception], List[Exception]]:
        """Retry only malformed/truncated segments after splitting them in half."""
        batch_results = extraction_chain.batch(
            segments,
            config={
                "max_concurrency": settings.LANGCHAIN_MAX_CONCURRENCY,
                "metadata": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "extraction_retry_depth": retry_depth,
                },
            },
            return_exceptions=True,
        )
        successful_results: List[GraphExtractionResult] = []
        terminal_errors: List[Exception] = []
        attempted_errors: List[Exception] = []

        for segment, result in zip(segments, batch_results):
            if isinstance(result, GraphExtractionResult):
                truncated = any(
                    diagnostic.get("diagnostic_type")
                    == "truncated_model_output"
                    for diagnostic in result.diagnostics
                )
                if not truncated:
                    successful_results.append(result)
                    continue
                result = GraphExtractionError(
                    "Extraction output reached its token limit",
                    model_responses=result.model_responses,
                )
            if not isinstance(result, Exception):
                result = RuntimeError(
                    "Extraction returned an unexpected segment result"
                )
            attempted_errors.append(result)

            retry_segments = [segment]
            if retry_depth < settings.EXTRACTION_SEGMENT_RETRY_DEPTH:
                retry_segments = self._split_failed_extraction_segment(segment)
            if len(retry_segments) <= 1:
                terminal_errors.append(result)
                continue

            logger.warning(
                "Retrying failed graph extraction segment: document_id=%s "
                "retry_depth=%d child_segments=%d error=%s",
                document_id,
                retry_depth + 1,
                len(retry_segments),
                str(result)[:500],
            )
            child_successes, child_terminal, child_attempts = (
                self._extract_segments_with_retry(
                    extraction_chain,
                    retry_segments,
                    document_id=document_id,
                    document_name=document_name,
                    retry_depth=retry_depth + 1,
                )
            )
            successful_results.extend(child_successes)
            terminal_errors.extend(child_terminal)
            attempted_errors.extend(child_attempts)

        return successful_results, terminal_errors, attempted_errors

    def _extract_document_graph(
        self,
        parsed_document: Any,
        document_title: str,
        document_id: str,
        file_hash: str = "",
        refresh_extraction: bool = False,
        cache_only: bool = False,
        document_name: str = "",
    ) -> GraphExtractionResult:
        """Extract a complete structure-aware document graph with versioned caching."""
        language = str(getattr(parsed_document, "language", "und") or "und")
        if refresh_extraction:
            logger.warning(
                "Refreshing extraction for document_id=%s bypasses the cache and may incur model charges (model=%s).",
                document_id,
                settings.EXTRACTION_MODEL,
            )
        cached = None if refresh_extraction else self._load_extraction_cache(
            document_id,
            file_hash,
            language=language,
        )
        self._raise_for_failed_cache(cached, document_id)
        if cached is not None and cached.cache_stage == "ready":
            logger.info("Using cached graph extraction for document_id=%s", document_id)
            return cached
        if cached is None and not refresh_extraction:
            logger.info(
                "Extraction cache miss for document_id=%s language=%s",
                document_id,
                language,
            )
        if cached is not None:
            if cache_only:
                raise RuntimeError(
                    f"Extraction cache for document {document_id} is incomplete; "
                    "graph rebuild will not call models."
                )
            with extraction_lock(file_hash or document_id):
                cached = self._load_extraction_cache(
                    document_id,
                    file_hash,
                    language=language,
                )
                self._raise_for_failed_cache(cached, document_id)
                if cached is None:
                    raise RuntimeError(
                        f"Extraction cache disappeared for document {document_id}"
                    )
                if cached.cache_stage == "ready":
                    return cached
                finalized = self._finalize_document_extraction(cached)
                self._save_extraction_cache(
                    document_id,
                    file_hash,
                    finalized,
                    language=language,
                )
                return finalized
        if cache_only:
            raise RuntimeError(
                f"No compatible extraction cache for document {document_id}; "
                "graph rebuild will not call an extraction model."
            )

        with extraction_lock(file_hash or document_id):
            cached = None if refresh_extraction else self._load_extraction_cache(
                document_id,
                file_hash,
                language=language,
            )
            self._raise_for_failed_cache(cached, document_id)
            if cached is not None and cached.cache_stage == "ready":
                logger.info(
                    "Using extraction cache populated by another worker: document_id=%s",
                    document_id,
                )
                return cached
            if cached is not None:
                if cache_only:
                    raise RuntimeError(
                        f"Extraction cache for document {document_id} is incomplete; "
                        "graph rebuild will not call models."
                    )
                finalized = self._finalize_document_extraction(cached)
                self._save_extraction_cache(
                    document_id,
                    file_hash,
                    finalized,
                    language=language,
                )
                return finalized
            if cache_only:
                raise RuntimeError(
                    f"No compatible extraction cache for document {document_id}; "
                    "graph rebuild will not call an extraction model."
                )

            structured_input = prepare_document_input(parsed_document)
            structured_input.document_id = document_id
            structured_input.title = document_title or structured_input.title
            if not structured_input.blocks:
                raise RuntimeError(
                    f"No usable source regions found for document {document_id}"
                )

            structured_input.spacy_candidates = {
                block.source_ref: spacy_candidate_extractor.extract(
                    block.text,
                    parsed_document.language,
                )
                for block in structured_input.blocks
            }
            vocabulary = self.graph_db.get_graph_vocabulary()
            structured_input.known_entity_types = vocabulary["entity_types"]
            structured_input.known_relationship_types = vocabulary[
                "relationship_types"
            ]
            candidate_names = [
                candidate["text"]
                for hints in structured_input.spacy_candidates.values()
                for candidate in hints.get("entities", [])
                if candidate.get("text")
            ]
            candidate_names.extend(
                phrase
                for hints in structured_input.spacy_candidates.values()
                for phrase in hints.get("noun_phrases", [])
                if phrase
            )
            structured_input.known_entities = self.graph_db.get_known_entity_context(
                candidate_names,
            )
            input_token_budget = (
                settings.EXTRACTION_CONTEXT_TOKENS
                - settings.EXTRACTION_OUTPUT_TOKENS
            )
            if input_token_budget < 1:
                raise ValueError(
                    "EXTRACTION_CONTEXT_TOKENS must exceed EXTRACTION_OUTPUT_TOKENS"
                )
            if settings.EXTRACTION_SEGMENT_MAX_CHARS < 1000:
                raise ValueError(
                    "EXTRACTION_SEGMENT_MAX_CHARS must be at least 1000"
                )
            if settings.EXTRACTION_SEGMENT_RETRY_DEPTH < 0:
                raise ValueError(
                    "EXTRACTION_SEGMENT_RETRY_DEPTH cannot be negative"
                )
            # Very large prompts fit the model context but lose facts in the
            # middle. Cap extraction regions by characters first, then enforce
            # the provider token budget. Ordinary documents remain one request.
            character_segments = split_document_input(
                structured_input,
                settings.EXTRACTION_SEGMENT_MAX_CHARS,
                len,
            )
            segments = [
                token_segment
                for character_segment in character_segments
                for token_segment in split_document_input(
                    character_segment,
                    input_token_budget,
                    lambda text: spacy_candidate_extractor.token_count(
                        text,
                        parsed_document.language,
                    ),
                )
            ]
            logger.info(
                "Prepared document graph extraction segments: document_id=%s "
                "segments=%d max_chars=%d",
                document_id,
                len(segments),
                settings.EXTRACTION_SEGMENT_MAX_CHARS,
            )
            extraction_chain = RunnableLambda(
                lambda item: entity_extractor.extract_document(item)
            ).with_config(
                {
                    "max_concurrency": settings.LANGCHAIN_MAX_CONCURRENCY,
                    "run_name": "document-graph-extraction",
                }
            )
            (
                segment_results,
                segment_errors,
                retry_attempt_errors,
            ) = self._extract_segments_with_retry(
                extraction_chain,
                segments,
                document_id=document_id,
                document_name=document_name,
            )
            if segment_errors:
                failed_extraction = self._merge_document_extractions(
                    segment_results,
                    document_id=document_id,
                )
                for error in retry_attempt_errors:
                    failed_extraction.model_responses.extend(
                        getattr(error, "model_responses", [])
                    )
                    failed_extraction.diagnostics.append(
                        {
                            "diagnostic_type": "graph_extraction_segment_failed",
                            "error_type": type(error).__name__,
                            "error": str(error)[:1000],
                        }
                    )
                if failed_extraction.model_responses:
                    failed_extraction.cache_stage = "failed"
                    failed_extraction.cache_metadata = {
                        "cache_format_version": CACHE_FORMAT_VERSION,
                        "raw_model_responses_available": True,
                    }
                    self._save_extraction_cache(
                        document_id,
                        file_hash,
                        failed_extraction,
                        language=language,
                    )
                raise RuntimeError(
                    f"Graph extraction failed for document {document_id}; "
                    "any model responses were cached for diagnosis. "
                    "Use --refresh-extraction to retry, which may incur model charges."
                ) from segment_errors[0]

            extracted = self._merge_document_extractions(
                segment_results,
                document_id=document_id,
            )
            for error in retry_attempt_errors:
                extracted.model_responses.extend(
                    getattr(error, "model_responses", [])
                )
                extracted.diagnostics.append(
                    {
                        "diagnostic_type": "graph_extraction_segment_retried",
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                )
            extracted.cache_metadata = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "raw_model_responses_available": bool(extracted.model_responses),
            }
            if extracted.document_type == "unknown":
                extracted.document_type = structured_input.document_type
            extracted.cache_stage = "raw"
            self._save_extraction_cache(
                document_id,
                file_hash,
                extracted,
                language=language,
            )
            resolved = self._finalize_document_extraction(extracted, vocabulary)
            self._save_extraction_cache(
                document_id,
                file_hash,
                resolved,
                language=language,
            )
            return resolved

    @staticmethod
    def _raise_for_failed_cache(
        cached: Optional[GraphExtractionResult],
        document_id: str,
    ) -> None:
        if cached is not None and cached.cache_stage == "failed":
            raise RuntimeError(
                f"Extraction cache for document {document_id} contains a failed "
                "model response. Inspect its diagnostics, then use "
                "--refresh-extraction to retry; this may incur model charges."
            )

    def _finalize_document_extraction(
        self,
        result: GraphExtractionResult,
        vocabulary: Optional[Dict[str, List[str]]] = None,
    ) -> GraphExtractionResult:
        """Normalize open types and resolve only high-confidence canonical matches."""
        graph_vocabulary = vocabulary or self.graph_db.get_graph_vocabulary()
        normalized = TypeNormalizer(embedding_model_client).normalize_result(
            result,
            existing_entity_types=graph_vocabulary["entity_types"],
            existing_relationship_types=graph_vocabulary["relationship_types"],
        )
        entities, diagnostics, _, id_map = EntityResolver(
            self.graph_db,
            self.vector_db,
        ).resolve(normalized.entities)
        normalized.entities = entities
        normalized.diagnostics.extend(diagnostics)

        relations: List[ExtractedRelation] = []
        for relation in normalized.relations:
            relation.source_entity_id = id_map.get(
                relation.source_entity_id,
                relation.source_entity_id,
            )
            relation.target_entity_id = id_map.get(
                relation.target_entity_id,
                relation.target_entity_id,
            )
            if not relation.source_entity_id or not relation.target_entity_id:
                normalized.diagnostics.append(
                    {
                        "diagnostic_type": "unresolved_relationship_endpoint",
                        "source": relation.source,
                        "target": relation.target,
                        "evidence": relation.evidence,
                        "confidence": relation.confidence,
                    }
                )
                continue
            relations.append(relation)
        normalized.relations = relations
        for mention in normalized.mentions:
            if mention.entity_id:
                mention.entity_id = id_map.get(mention.entity_id, mention.entity_id)
        normalized.cache_stage = "ready"
        return normalized

    @staticmethod
    def _merge_document_extractions(
        results: List[GraphExtractionResult],
        document_id: str,
    ) -> GraphExtractionResult:
        """Resolve segment-local IDs and merge duplicated evidence conservatively."""
        registry = DocumentEntityRegistry(document_id=document_id)
        diagnostics: List[Dict[str, Any]] = []
        mentions: Dict[str, ExtractedMention] = {}
        relations: Dict[tuple[str, str, str, str], ExtractedRelation] = {}
        topics_by_name: Dict[str, ExtractedTopic] = {}
        document_type = "unknown"

        for result in results:
            if result.document_type != "unknown":
                if document_type == "unknown":
                    document_type = result.document_type
                elif document_type.casefold() != result.document_type.casefold():
                    diagnostics.append(
                        {
                            "diagnostic_type": "document_type_conflict",
                            "types": [document_type, result.document_type],
                        }
                    )
            diagnostics.extend(result.diagnostics)
            for entity in result.entities:
                registry.register(entity)
            for topic in result.topics:
                key = " ".join(topic.name.casefold().split())
                current = topics_by_name.get(key)
                if current is None:
                    topics_by_name[key] = topic
                else:
                    current.aliases = sorted(set(current.aliases) | set(topic.aliases))
                    current.parent_names = list(
                        dict.fromkeys(current.parent_names + topic.parent_names)
                    )
                    if topic.confidence > current.confidence:
                        current.evidence = topic.evidence
                        current.confidence = topic.confidence
                        current.source_ref = topic.source_ref
                        current.provenance = topic.provenance
            for mention in result.mentions:
                mention.entity_id = registry.resolve(
                    entity_id=mention.entity_id,
                    name=mention.entity_name,
                    category=mention.entity_category,
                ) or ""
                mention.mention_id = hashlib.sha256(
                    f"{document_id}|{mention.text}|{mention.entity_id}|{mention.evidence}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
                mentions[mention.mention_id] = mention
            for relation in result.relations:
                relation.source_entity_id = registry.resolve(
                    entity_id=relation.source_entity_id,
                    name=relation.source,
                    category=relation.source_category,
                ) or ""
                relation.target_entity_id = registry.resolve(
                    entity_id=relation.target_entity_id,
                    name=relation.target,
                    category=relation.target_category,
                ) or ""
                if not relation.source_entity_id or not relation.target_entity_id:
                    diagnostics.append(
                        {
                            "diagnostic_type": "unresolved_relationship_endpoint",
                            "source": relation.source,
                            "target": relation.target,
                            "evidence": relation.evidence,
                            "confidence": relation.confidence,
                        }
                    )
                    continue
                key = (
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation.relation_type,
                    hashlib.sha256(relation.evidence.encode("utf-8")).hexdigest()[:16],
                )
                relations[key] = relation

        entity_names = {
            normalize_topic(value)
            for entity in registry.entities.values()
            for value in [entity.name, *entity.aliases]
            if value
        }
        accepted_topics: List[ExtractedTopic] = []
        for topic in topics_by_name.values():
            normalized = normalize_topic(topic.name)
            if normalized in entity_names:
                diagnostics.append(
                    {
                        "diagnostic_type": "topic_matches_entity",
                        "name": topic.name,
                        "evidence": topic.evidence,
                        "confidence": topic.confidence,
                    }
                )
                continue
            accepted_topics.append(topic)

        topic_names: Dict[str, List[str]] = {}
        for topic in accepted_topics:
            for name in [topic.name, *topic.aliases]:
                topic_names.setdefault(" ".join(name.casefold().split()), []).append(topic.name)
        for topic in accepted_topics:
            parents: List[str] = []
            for parent in topic.parent_names:
                candidates = list(
                    dict.fromkeys(topic_names.get(" ".join(parent.casefold().split()), []))
                )
                if len(candidates) == 1 and candidates[0] != topic.name:
                    parents.append(candidates[0])
                else:
                    diagnostics.append(
                        {
                            "diagnostic_type": "ambiguous_topic_parent",
                            "name": topic.name,
                            "parent": parent,
                            "evidence": topic.evidence,
                            "confidence": topic.confidence,
                        }
                    )
            topic.parent_names = list(dict.fromkeys(parents))

        return GraphExtractionResult(
            entities=list(registry.entities.values()),
            mentions=list(mentions.values()),
            relations=list(relations.values()),
            topics=accepted_topics,
            diagnostics=diagnostics,
            model_responses=[
                response
                for result in results
                for response in result.model_responses
            ],
            document_type=document_type,
        )

    @staticmethod
    def _cache_path(document_id: str) -> Path:
        key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        return settings.EXTRACTION_CACHE_DIR / f"{key}.json"

    @staticmethod
    def _cache_signature(language: str = "und") -> Dict[str, str]:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "model": settings.EXTRACTION_MODEL,
            "provider": settings.MODEL_API_BASE_URL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "language": (language or "und").casefold(),
            "prompt_version": settings.EXTRACTION_PROMPT_VERSION,
            "pipeline_version": settings.EXTRACTION_VERSION,
            "normalization_version": settings.NORMALIZATION_VERSION,
            "evidence_match_version": EVIDENCE_MATCH_VERSION,
            "context_tokens": str(settings.EXTRACTION_CONTEXT_TOKENS),
            "output_tokens": str(settings.EXTRACTION_OUTPUT_TOKENS),
            "segment_max_characters": str(settings.EXTRACTION_SEGMENT_MAX_CHARS),
            "segment_retry_depth": str(settings.EXTRACTION_SEGMENT_RETRY_DEPTH),
            "minimum_confidence": str(settings.EXTRACTION_MIN_CONFIDENCE),
            "topic_min_confidence": str(settings.TOPIC_MIN_CONFIDENCE),
            "spacy_models": "|".join(
                [
                    str(getattr(spacy_candidate_extractor, "english_model", "")),
                    str(
                        getattr(
                            spacy_candidate_extractor,
                            "multilingual_parser_model",
                            "",
                        )
                    ),
                    str(
                        getattr(
                            spacy_candidate_extractor,
                            "multilingual_ner_model",
                            "",
                        )
                    ),
                ]
            ),
        }

    @staticmethod
    def _legacy_cache_signature() -> Dict[str, str]:
        return {
            "model": settings.EXTRACTION_MODEL,
            "prompt_version": settings.EXTRACTION_PROMPT_VERSION,
            "pipeline_version": settings.EXTRACTION_VERSION,
            "normalization_version": settings.NORMALIZATION_VERSION,
        }

    def _load_extraction_cache(
        self,
        document_id: str,
        file_hash: str,
        *,
        language: str = "und",
    ) -> Optional[GraphExtractionResult]:
        path = self._cache_path(document_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached_file_hash = payload.get("file_hash")
            if not cached_file_hash:
                raise ValueError("Cache record is missing its document content hash")
            if file_hash and cached_file_hash != file_hash:
                return None
            cache_signature = payload.get("cache_signature")
            is_legacy_cache = cache_signature == self._legacy_cache_signature()
            if cache_signature != self._cache_signature(language):
                if is_legacy_cache:
                    logger.warning(
                        "Using legacy extraction cache for document_id=%s; it lacks raw model responses and language/configuration metadata. Use --refresh-extraction to regenerate it, which may incur model charges.",
                        document_id,
                    )
                else:
                    raise RuntimeError(
                        f"Extraction cache for document {document_id} is incompatible "
                        "with the current language, model, or extraction settings. "
                        "Use --refresh-extraction to replace it; this may incur model charges."
                    )
            data = payload["result"]
            cache_metadata = data.get("cache_metadata")
            if not isinstance(cache_metadata, dict):
                cache_metadata = {}
            if is_legacy_cache:
                cache_metadata = {
                    **cache_metadata,
                    "legacy_cache": True,
                    "raw_model_responses_available": False,
                }
            return GraphExtractionResult(
                entities=[ExtractedEntity(**row) for row in data.get("entities", [])],
                mentions=[ExtractedMention(**row) for row in data.get("mentions", [])],
                relations=[ExtractedRelation(**row) for row in data.get("relations", [])],
                topics=[ExtractedTopic(**row) for row in data.get("topics", [])],
                diagnostics=data.get("diagnostics", []),
                model_responses=data.get("model_responses", []),
                cache_metadata=cache_metadata,
                document_type=data.get("document_type", "unknown"),
                cache_stage=data.get("cache_stage", "ready"),
            )
        except RuntimeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RuntimeError(
                f"Extraction cache for document {document_id} is invalid. "
                "Use --refresh-extraction to replace it; this may incur model charges."
            ) from exc

    def _save_extraction_cache(
        self,
        document_id: str,
        file_hash: str,
        result: GraphExtractionResult,
        *,
        language: str = "und",
    ) -> None:
        path = self._cache_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "document_id": document_id,
            "file_hash": file_hash,
            "cache_signature": self._cache_signature(language),
            "result": asdict(result),
        }
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)

    def check_cross_store_consistency(self) -> Dict[str, Any]:
        """Verify consistency between ChromaDB vector chunks and Neo4j graph document nodes."""
        chroma_docs = vector_store.get_indexed_documents()
        neo4j_docs = graph_builder.get_indexed_documents()

        with get_db_connection() as conn:
            journal_rows = conn.execute(
                """
                SELECT document_name, file_hash, status, stage, updated_at
                FROM ingestion_runs
                WHERE status != 'completed'
                ORDER BY updated_at DESC
                """
            ).fetchall()
            incomplete_runs = [dict(r) for r in journal_rows]

        in_chroma_only = sorted(list(chroma_docs - neo4j_docs)) if neo4j_docs else []
        in_neo4j_only = sorted(list(neo4j_docs - chroma_docs)) if neo4j_docs else []
        is_consistent = (len(in_chroma_only) == 0 and len(in_neo4j_only) == 0 and len(incomplete_runs) == 0)

        report = {
            "consistent": is_consistent,
            "chroma_document_count": len(chroma_docs),
            "neo4j_document_count": len(neo4j_docs),
            "incomplete_runs_count": len(incomplete_runs),
            "incomplete_runs": incomplete_runs,
            "orphaned_in_chroma": in_chroma_only,
            "orphaned_in_neo4j": in_neo4j_only,
        }
        if not is_consistent:
            logger.warning("Cross-store consistency discrepancy: %s", report)
        return report

    # ── Mode B Helper: Archive Upload Handler ──
    def ingest_uploaded_archive(
        self,
        archive_bytes: bytes,
        archive_name: str,
        user_id: Optional[int] = None,
    ) -> List[IngestionResult]:
        """
        Unpack an archive in a secure temporary sandbox and process all contained files.
        Cleans up sandbox after execution.
        """
        results: List[IngestionResult] = []
        try:
            extracted_files = archive_handler.extract(archive_bytes, archive_name=archive_name)
            for ef in extracted_files:
                with open(ef.path, "rb") as f:
                    file_bytes = f.read()
                res = self.ingest_uploaded_file(
                    file_bytes=file_bytes,
                    filename=ef.filename,
                    user_id=user_id,
                    source_archive=archive_name,
                )
                results.append(res)
        finally:
            # Clean up all temp directories in UPLOAD_TEMP_DIR
            for sandbox in settings.UPLOAD_TEMP_DIR.glob("sandbox_*"):
                archive_handler.cleanup_sandbox(sandbox)

        return results


pipeline = IngestionPipeline()
