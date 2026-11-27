"""Local structured-file parser checks; uses temporary configuration and fixtures."""

import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from graph_rag.config import settings


class _MockSpacyDoc:
    def __init__(
        self,
        text: str,
        *,
        sentences: list[str] | None = None,
        tokens: list[str] | None = None,
        entities: list[object] | None = None,
        noun_chunks: list[str] | None = None,
    ) -> None:
        from types import SimpleNamespace

        self.text = text
        self.sents = [SimpleNamespace(text=value) for value in (sentences or [])]
        self.ents = entities or []
        self._noun_chunks = noun_chunks
        self._tokens = [
            SimpleNamespace(
                text=value,
                is_space=False,
                dep_="ROOT",
                head=SimpleNamespace(text=value),
            )
            for value in (tokens or [])
        ]

    def __iter__(self):
        return iter(self._tokens)

    @property
    def noun_chunks(self):
        if self._noun_chunks is None:
            raise AssertionError("noun_chunks accessed without a dependency parser")
        return [type("MockPhrase", (), {"text": value})() for value in self._noun_chunks]


class _MockSpacyModel:
    def __init__(self, name: str, pipe_names: list[str], doc: _MockSpacyDoc) -> None:
        self.meta = {"name": name}
        self.pipe_names = pipe_names
        self.doc = doc

    def __call__(self, text: str) -> _MockSpacyDoc:
        return self.doc

    def make_doc(self, text: str) -> _MockSpacyDoc:
        return self.doc


class StructuredFileParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._temporary_root = Path(cls._temporary_directory.name)
        setting_paths = {
            "DATA_DIR": cls._temporary_root / "data",
            "PDF_DIR": cls._temporary_root / "documents",
            "METADATA_FILE": cls._temporary_root / "metadata.csv",
            "CHROMA_PERSIST_DIR": cls._temporary_root / "chroma",
            "SQLITE_DB_PATH": cls._temporary_root / "data" / "test.sqlite",
            "UPLOAD_TEMP_DIR": cls._temporary_root / "uploads",
            "EXTRACTION_CACHE_DIR": cls._temporary_root / "cache",
        }
        cls._original_paths = {
            name: getattr(settings, name) for name in setting_paths
        }
        for name, value in setting_paths.items():
            object.__setattr__(settings, name, value)
        settings.ensure_directories()
        from graph_rag.ingestion.file_handler import FileHandler

        cls.handler = FileHandler()

    @classmethod
    def tearDownClass(cls) -> None:
        for name, value in cls._original_paths.items():
            object.__setattr__(settings, name, value)
        cls._temporary_directory.cleanup()

    def test_arabic_parquet_is_supported_and_keeps_rows(self) -> None:
        import pandas as pd

        parquet = io.BytesIO()
        pd.DataFrame(
            [{"العنوان": "بيانات عربية", "القيمة": 42}]
        ).to_parquet(parquet, index=False)

        document = self.handler.parse(
            parquet.getvalue(),
            filename="arabic.parquet",
        )

        self.assertIn("بيانات عربية", document.full_text)
        self.assertEqual(document.table_data[0]["rows"][0]["العنوان"], "بيانات عربية")
        self.assertEqual(document.language, "ar")
        self.assertEqual(document.file_type, "parquet")
        self.assertEqual(document.document_id, document.content_hash)

    def test_spacy_sentence_only_pipeline_keeps_sentences_and_tokens(self) -> None:
        from unittest.mock import patch
        from graph_rag.ingestion.spacy_candidates import SpacyCandidateExtractor

        text = "First sentence. Second sentence."
        tokenizer_doc = _MockSpacyDoc(
            text,
            sentences=["First sentence.", "Second sentence."],
            tokens=["First", "sentence", ".", "Second", "sentence", "."],
        )
        tokenizer = _MockSpacyModel("sentence-only", ["senter"], tokenizer_doc)
        ner = _MockSpacyModel(
            "without-ner",
            [],
            _MockSpacyDoc(text),
        )
        extractor = SpacyCandidateExtractor()

        with (
            patch.object(extractor, "_models_for", return_value=(tokenizer, ner)),
            self.assertLogs("graph_rag.ingestion.spacy_candidates", level="WARNING") as logs,
        ):
            candidates = extractor.extract(text, "ar")
            extractor.extract(text, "ar")

        self.assertEqual(candidates["sentences"], ["First sentence.", "Second sentence."])
        self.assertEqual(candidates["tokens"], ["First", "sentence", ".", "Second", "sentence", "."])
        self.assertEqual(candidates["dependencies"], [])
        self.assertEqual(candidates["entities"], [])
        self.assertEqual(len(logs.records), 3)
        self.assertTrue(
            any("Arabic dependency parsing" in record.getMessage() for record in logs.records)
        )

    def test_spacy_ner_only_pipeline_keeps_ner_and_uses_other_fallbacks(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        from graph_rag.ingestion.spacy_candidates import SpacyCandidateExtractor

        text = "Khaled met Microsoft."
        tokenizer_doc = _MockSpacyDoc(
            text,
            tokens=["Khaled", "met", "Microsoft", "."],
        )
        ner_doc = _MockSpacyDoc(
            text,
            entities=[
                SimpleNamespace(
                    text="Microsoft",
                    label_="ORG",
                    start_char=11,
                    end_char=20,
                )
            ],
        )
        tokenizer = _MockSpacyModel("tokenizer", [], tokenizer_doc)
        ner = _MockSpacyModel("ner-only", ["ner"], ner_doc)
        extractor = SpacyCandidateExtractor()

        with (
            patch.object(extractor, "_models_for", return_value=(tokenizer, ner)),
            self.assertLogs("graph_rag.ingestion.spacy_candidates", level="WARNING") as logs,
        ):
            candidates = extractor.extract(text, "ar")

        self.assertEqual(candidates["sentences"], [text])
        self.assertEqual(candidates["tokens"], ["Khaled", "met", "Microsoft", "."])
        self.assertEqual(candidates["dependencies"], [])
        self.assertEqual(candidates["entities"][0]["text"], "Microsoft")
        self.assertEqual(len(logs.records), 3)
        self.assertTrue(
            any("Arabic dependency parsing" in record.getMessage() for record in logs.records)
        )

    def test_spacy_full_pipeline_uses_all_available_capabilities(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        from graph_rag.ingestion.spacy_candidates import SpacyCandidateExtractor

        text = "Khaled joined Microsoft."
        doc = _MockSpacyDoc(
            text,
            sentences=[text],
            tokens=["Khaled", "joined", "Microsoft", "."],
            entities=[
                SimpleNamespace(
                    text="Microsoft",
                    label_="ORG",
                    start_char=14,
                    end_char=23,
                )
            ],
            noun_chunks=["Khaled", "Microsoft"],
        )
        model = _MockSpacyModel("full", ["parser", "ner"], doc)
        extractor = SpacyCandidateExtractor()

        with patch.object(extractor, "_models_for", return_value=(model, model)):
            candidates = extractor.extract(text, "en")

        self.assertEqual(candidates["sentences"], [text])
        self.assertEqual(candidates["tokens"], ["Khaled", "joined", "Microsoft", "."])
        self.assertEqual(
            [dependency["dependency"] for dependency in candidates["dependencies"]],
            ["ROOT"] * 4,
        )
        self.assertEqual(candidates["entities"][0]["text"], "Microsoft")
        self.assertEqual(candidates["noun_phrases"], ["Khaled", "Microsoft"])

    def test_spacy_runtime_failure_uses_non_entity_lexical_fallback(self) -> None:
        import builtins
        from graph_rag.ingestion.spacy_candidates import SpacyCandidateExtractor

        extractor = SpacyCandidateExtractor()
        original_import = builtins.__import__

        def block_spacy_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("mock blocked runtime")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=block_spacy_import),
            self.assertLogs("graph_rag.ingestion.spacy_candidates", level="WARNING") as logs,
        ):
            candidates = extractor.extract(
                "Khaled met Microsoft.\nA new line.",
                "en",
            )
            token_count = extractor.token_count("Khaled met Microsoft.", "en")

        self.assertEqual(candidates["sentences"], ["Khaled met Microsoft.", "A new line."])
        self.assertEqual(candidates["entities"], [])
        self.assertEqual(candidates["dependencies"], [])
        self.assertEqual(candidates["noun_phrases"], [])
        self.assertEqual(
            token_count,
            sum(not character.isspace() for character in "Khaled met Microsoft."),
        )
        self.assertEqual(len(logs.records), 1)

    def test_csv_json_and_excel_keep_searchable_rows(self) -> None:
        import pandas as pd

        csv_doc = self.handler.parse(
            "name,value\nArabic,1\n".encode("utf-8"),
            filename="table.csv",
        )
        json_doc = self.handler.parse(
            '{"name":"Arabic","value":2}'.encode("utf-8"),
            filename="table.json",
        )
        excel = io.BytesIO()
        pd.DataFrame([{"name": "Arabic", "value": 3}]).to_excel(
            excel, index=False, sheet_name="Records"
        )
        xlsx_doc = self.handler.parse(excel.getvalue(), filename="table.xlsx")

        self.assertIn('"name": "Arabic"', csv_doc.full_text)
        self.assertIn('"name": "Arabic"', json_doc.full_text)
        self.assertEqual(xlsx_doc.pages[0].sheet_name, "Records")
        self.assertEqual(xlsx_doc.table_data[0]["rows"][0]["value"], 3)

    def test_scanned_pdf_uses_mock_ocr_and_preserves_page_boxes(self) -> None:
        import pymupdf
        from PIL import Image
        from graph_rag.ingestion.ocr import OCRResult
        from graph_rag.ingestion.pdf_parser import PDFParser

        class FakeOCRProvider:
            def __init__(self) -> None:
                self.calls = []

            def extract(self, image_bytes, language):
                self.calls.append((image_bytes, language))
                return OCRResult(
                    text="Arabic scanned evidence",
                    bounding_boxes=[
                        {
                            "text": "Arabic",
                            "x": 20,
                            "y": 40,
                            "width": 60,
                            "height": 20,
                            "confidence": 0.9,
                            "coordinate_unit": "pixel",
                        }
                    ],
                    confidence=0.9,
                )

        image_buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(image_buffer, format="PNG")
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_image(
            pymupdf.Rect(20, 20, 200, 200),
            stream=image_buffer.getvalue(),
        )
        pdf_bytes = pdf.tobytes()
        pdf.close()

        provider = FakeOCRProvider()
        document = PDFParser(
            ocr_provider=provider,
            ocr_dpi=200,
            ocr_languages="eng+ara",
        ).parse(pdf_bytes, filename="scan.pdf")

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][1], "eng+ara")
        self.assertIn("Arabic scanned evidence", document.full_text)
        self.assertEqual(document.pages[0].page_number, 1)
        self.assertEqual(document.pages[0].bounding_boxes[0]["coordinate_unit"], "point")
        self.assertAlmostEqual(document.pages[0].bounding_boxes[0]["x"], 7.2)

    def test_pdf_tables_are_retained_as_rows(self) -> None:
        import pymupdf
        from graph_rag.ingestion.pdf_parser import PDFParser

        pdf = pymupdf.open()
        page = pdf.new_page()
        x_positions = [72, 220, 360]
        y_positions = [72, 112, 152]
        for x in x_positions:
            page.draw_line((x, y_positions[0]), (x, y_positions[-1]), width=1)
        for y in y_positions:
            page.draw_line((x_positions[0], y), (x_positions[-1], y), width=1)
        page.insert_text((80, 98), "Name")
        page.insert_text((228, 98), "Value")
        page.insert_text((80, 138), "Dataset")
        page.insert_text((228, 138), "Research")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        document = PDFParser().parse(pdf_bytes, filename="table.pdf")

        self.assertEqual(len(document.table_data), 1)
        self.assertIn("Name", document.table_data[0]["columns"])
        self.assertTrue(
            any("Dataset" in row.values() for row in document.table_data[0]["rows"])
        )

    def test_text_pdf_routes_embedded_images_to_ocr(self) -> None:
        import pymupdf
        from PIL import Image
        from graph_rag.ingestion.ocr import OCRResult
        from graph_rag.ingestion.pdf_parser import PDFParser

        class FakeOCRProvider:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, image_bytes, language):
                self.calls += 1
                return OCRResult(
                    text="Figure label",
                    bounding_boxes=[
                        {
                            "text": "Figure",
                            "x": 4,
                            "y": 5,
                            "width": 20,
                            "height": 10,
                            "confidence": 0.8,
                        }
                    ],
                    confidence=0.8,
                )

        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(image_buffer, format="PNG")
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "This PDF page has enough embedded text to avoid page-level OCR.",
        )
        page.insert_image(
            pymupdf.Rect(100, 100, 200, 200),
            stream=image_buffer.getvalue(),
        )
        pdf_bytes = pdf.tobytes()
        pdf.close()
        provider = FakeOCRProvider()

        document = PDFParser(ocr_provider=provider).parse(
            pdf_bytes,
            filename="text-with-image.pdf",
        )

        self.assertEqual(provider.calls, 1)
        self.assertIn("Figure label", document.pages[0].ocr_text)
        self.assertIn("Image OCR", document.pages[0].text)

    def test_arabic_verification_ocr_replaces_corrupted_embedded_text(self) -> None:
        import pymupdf
        from graph_rag.ingestion.ocr import OCRResult
        from graph_rag.ingestion.pdf_parser import PDFParser

        class FakeOCRProvider:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, image_bytes, language):
                self.calls += 1
                return OCRResult(
                    text="يشهد الاقتصاد العالمي تحولات كبيرة في السنوات الأخيرة",
                    confidence=0.91,
                )

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "Corrupted embedded text that is long enough to pass the old length check.",
        )
        pdf_bytes = pdf.tobytes()
        pdf.close()
        provider = FakeOCRProvider()

        with patch.object(PDFParser, "_needs_full_page_ocr", return_value=True):
            document = PDFParser(ocr_provider=provider).parse(
                pdf_bytes,
                filename="arabic-report.pdf",
            )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(document.pages[0].extraction_method, "tesseract_ocr")
        self.assertEqual(
            document.pages[0].text,
            "يشهد الاقتصاد العالمي تحولات كبيرة في السنوات الأخيرة",
        )

    def test_pdf_quality_checks_arabic_and_generic_office_title(self) -> None:
        from graph_rag.ingestion.pdf_parser import PDFParser

        self.assertTrue(
            PDFParser._needs_full_page_ocr(
                "يشهد الاقتصاد العالمي تحولات كبيرة في السنوات الأخيرة"
            )
        )
        self.assertTrue(PDFParser._is_generic_title("PowerPoint Presentation"))
        self.assertFalse(PDFParser._is_generic_title("آفاق اقتصادية العدد 57"))

    def test_image_parser_uses_replaceable_ocr_provider(self) -> None:
        from graph_rag.ingestion.image_parser import ImageParser
        from graph_rag.ingestion.ocr import OCRResult

        class FakeOCRProvider:
            def extract(self, image_bytes, language):
                self.language = language
                return OCRResult(text="Mixed English نص عربي", confidence=0.8)

        provider = FakeOCRProvider()
        document = ImageParser(
            ocr_provider=provider,
            ocr_languages="eng+ara",
        ).parse(b"fake image bytes", filename="figure.png")

        self.assertEqual(provider.language, "eng+ara")
        self.assertEqual(document.language, "mixed")
        self.assertEqual(document.pages[0].ocr_text, document.full_text)

    def test_tesseract_adapter_returns_text_confidence_and_boxes(self) -> None:
        from unittest.mock import patch
        from PIL import Image
        from graph_rag.ingestion.ocr import TesseractOCRProvider

        image_buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(image_buffer, format="PNG")
        with patch(
            "pytesseract.image_to_data",
            return_value={
                "text": ["مرحبا", "world", ""],
                "conf": ["90", "80", "-1"],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 1, 1],
                "left": [1, 10, 0],
                "top": [2, 2, 0],
                "width": [8, 8, 0],
                "height": [4, 4, 0],
            },
        ) as ocr_call:
            result = TesseractOCRProvider().extract(
                image_buffer.getvalue(),
                language="eng+ara",
            )

        self.assertIn("مرحبا world", result.text)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertEqual(ocr_call.call_args.kwargs["lang"], "eng+ara")

    def test_docx_table_rows_are_retained(self) -> None:
        from docx import Document
        from graph_rag.ingestion.docx_parser import DocxParser

        docx = Document()
        table = docx.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Arabic"
        table.cell(1, 1).text = "بيانات"
        buffer = io.BytesIO()
        docx.save(buffer)

        parsed = DocxParser().parse(buffer.getvalue(), filename="table.docx")

        self.assertEqual(parsed.table_data[0]["columns"], ["Name", "Value"])
        self.assertEqual(
            parsed.table_data[0]["rows"][0]["Value"],
            "بيانات",
        )
        self.assertIn("بيانات", parsed.full_text)

    def test_pptx_table_rows_are_retained(self) -> None:
        from pptx import Presentation
        from pptx.util import Inches
        from graph_rag.ingestion.pptx_parser import PptxParser

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        table = slide.shapes.add_table(
            2,
            2,
            Inches(1),
            Inches(1),
            Inches(4),
            Inches(1),
        ).table
        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Arabic"
        table.cell(1, 1).text = "بيانات"
        buffer = io.BytesIO()
        presentation.save(buffer)

        parsed = PptxParser().parse(buffer.getvalue(), filename="table.pptx")

        self.assertEqual(parsed.table_data[0]["columns"], ["Name", "Value"])
        self.assertEqual(
            parsed.table_data[0]["rows"][0]["Value"],
            "بيانات",
        )

    def test_docx_and_pptx_images_route_through_ocr_provider(self) -> None:
        from PIL import Image
        from docx import Document
        from docx.shared import Inches as DocxInches
        from pptx import Presentation
        from pptx.util import Inches as PptxInches
        from graph_rag.ingestion.docx_parser import DocxParser
        from graph_rag.ingestion.ocr import OCRResult
        from graph_rag.ingestion.pptx_parser import PptxParser

        class FakeOCRProvider:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, image_bytes, language):
                self.calls += 1
                return OCRResult(
                    text="Image label",
                    bounding_boxes=[{"text": "Image", "x": 1, "y": 2}],
                    confidence=0.75,
                )

        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(image_buffer, format="PNG")

        docx = Document()
        docx.add_picture(io.BytesIO(image_buffer.getvalue()), width=DocxInches(1))
        docx_buffer = io.BytesIO()
        docx.save(docx_buffer)
        docx_ocr = FakeOCRProvider()
        parsed_docx = DocxParser(ocr_provider=docx_ocr).parse(
            docx_buffer.getvalue(),
            filename="image.docx",
        )

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_picture(
            io.BytesIO(image_buffer.getvalue()),
            PptxInches(1),
            PptxInches(1),
        )
        pptx_buffer = io.BytesIO()
        presentation.save(pptx_buffer)
        pptx_ocr = FakeOCRProvider()
        parsed_pptx = PptxParser(ocr_provider=pptx_ocr).parse(
            pptx_buffer.getvalue(),
            filename="image.pptx",
        )

        self.assertEqual(docx_ocr.calls, 1)
        self.assertEqual(pptx_ocr.calls, 1)
        self.assertIn("Image label", parsed_docx.full_text)
        self.assertIn("Image label", parsed_pptx.full_text)

    def test_parser_registry_accepts_additional_extensions(self) -> None:
        from graph_rag.ingestion.pdf_parser import ParsedDocument

        handler = type(self.handler)()
        handler.register_parser(
            ".custom",
            lambda source, filename: ParsedDocument(
                filename=filename,
                title="Custom",
                total_pages=0,
                full_text="",
            ),
        )

        self.assertTrue(handler.is_supported("input.custom"))
        self.assertEqual(handler.parse(b"", "input.custom").title, "Custom")

    def test_content_hash_deduplication_survives_renaming(self) -> None:
        import hashlib

        from graph_rag.ingestion.dedup_checker import DedupChecker
        from graph_rag.ingestion.metadata_manager import MetadataManager

        content = b"arabic parquet source"
        manager = MetadataManager(self._temporary_root / "dedup.csv")
        manager.append_document(
            {
                "local_pdf": "original.parquet",
                "file_hash": hashlib.sha256(content).hexdigest(),
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
        )

        result = DedupChecker(manager).check_bytes(content, "renamed.parquet")

        self.assertTrue(result.is_duplicate)
        self.assertEqual(result.match_type, "hash")

    def test_structured_rows_are_saved_in_additive_sqlite_schema(self) -> None:
        from graph_rag.database.db import (
            get_db_connection,
            init_db,
            save_document_pages,
            save_structured_tables,
        )
        from graph_rag.ingestion.pdf_parser import PageContent

        init_db()
        save_structured_tables(
            "arabic.parquet",
            "content-hash",
            [
                {
                    "table_name": "data",
                    "columns": ["العنوان"],
                    "rows": [{"العنوان": "بيانات"}],
                }
            ],
        )
        save_document_pages(
            "arabic.parquet",
            "content-hash",
            [
                PageContent(
                    page_number=1,
                    text="بيانات",
                    ocr_text="نص",
                    source_offsets=[{"start": 0, "end": 6}],
                    bounding_boxes=[{"x": 1, "y": 2, "text": "بيانات"}],
                    extraction_method="tesseract_ocr",
                    confidence=0.9,
                    language="ar",
                )
            ],
        )

        with get_db_connection() as connection:
            row = connection.execute(
                "SELECT document_name, file_hash, rows_json "
                "FROM structured_tables WHERE document_name = ?",
                ("arabic.parquet",),
            ).fetchone()
            page_row = connection.execute(
                "SELECT language, ocr_text, bounding_boxes_json "
                "FROM document_pages WHERE document_name = ?",
                ("arabic.parquet",),
            ).fetchone()

        self.assertEqual(row["file_hash"], "content-hash")
        self.assertIn("بيانات", row["rows_json"])
        self.assertEqual(page_row["language"], "ar")
        self.assertIn("نص", page_row["ocr_text"])
        self.assertIn("بيانات", page_row["bounding_boxes_json"])

    def test_graph_initializer_only_adds_constraints(self) -> None:
        from graph_rag.graph.graph_builder import GraphBuilder

        class FakeResult:
            def consume(self):
                return self

        class FakeSession:
            def __init__(self):
                self.queries = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def run(self, query, **params):
                self.queries.append(query)
                return FakeResult()

        class FakeDriver:
            def __init__(self):
                self.session_instance = FakeSession()

            def session(self):
                return self.session_instance

        builder = GraphBuilder()
        driver = FakeDriver()
        builder._driver = driver

        builder.init_constraints()

        self.assertEqual(len(driver.session_instance.queries), 4)
        self.assertTrue(
            all(
                "DROP" not in query.upper() and "DELETE" not in query.upper()
                for query in driver.session_instance.queries
            )
        )

    def test_arabic_normalization_and_query_classification(self) -> None:
        from graph_rag.graph.topic_taxonomy import normalize_topic
        from graph_rag.ingestion.dedup_checker import normalize_title

        classifier_path = (
            PROJECT_SRC / "graph_rag" / "retriever" / "query_classifier.py"
        )
        spec = importlib.util.spec_from_file_location(
            "isolated_query_classifier",
            classifier_path,
        )
        classifier_module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = classifier_module
        spec.loader.exec_module(classifier_module)
        query_classifier = classifier_module.query_classifier

        arabic = "معالجة اللغة الطبيعية"
        self.assertEqual(normalize_topic(arabic), arabic)
        self.assertEqual(normalize_title(arabic), arabic)

        factual = query_classifier.classify("ما هي معالجة اللغة الطبيعية؟")
        self.assertEqual(
            factual.query_class,
            classifier_module.QueryClass.FACTUAL_LOOKUP,
        )
        self.assertNotIn("ما", factual.extracted_entities)

    def test_detects_mixed_language_text(self) -> None:
        document = self.handler.parse(
            "English text مع نص عربي".encode("utf-8"),
            filename="mixed.txt",
        )

        self.assertEqual(document.language, "mixed")

    def test_chunker_uses_langchain_splitter_with_page_offsets(self) -> None:
        from graph_rag.ingestion.chunker import TextChunker
        from graph_rag.ingestion.pdf_parser import PageContent, ParsedDocument

        text = "one two three four five six seven eight nine ten eleven twelve"
        document = ParsedDocument(
            filename="split.txt",
            title="Split",
            total_pages=1,
            full_text=text,
            pages=[PageContent(page_number=4, text=text)],
        )
        document.document_id = "document-id"
        document.content_hash = "content-hash"
        document.file_type = "txt"
        document.mime_type = "text/plain"
        document.language = "ar"

        chunks = TextChunker(chunk_size=8, chunk_overlap=2).chunk_document(document)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 4)
        self.assertEqual(chunks[0].metadata["start_index"], 0)
        self.assertEqual(chunks[0].metadata["content_hash"], "content-hash")
        self.assertEqual(chunks[0].metadata["language"], "ar")
        self.assertTrue(all(chunk.metadata["start_index"] >= 0 for chunk in chunks))

    def test_chat_client_invokes_a_langchain_runnable_without_network(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        from graph_rag.llm.model_client import CloudModelClient

        class FakeChatModel:
            def bind(self, **kwargs):
                return RunnableLambda(
                    lambda prompt_value: self._respond(prompt_value)
                )

            @staticmethod
            def _respond(prompt_value):
                messages = prompt_value.to_messages()
                assert "What is in this source?" in messages[-1].content
                return AIMessage(
                    content="Grounded response",
                    usage_metadata={
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "total_tokens": 11,
                    },
                )

        client = CloudModelClient(
            api_key="mock-key",
            llm_model="mock-model",
            role="test",
        )
        client._get_chat_model = lambda: FakeChatModel()
        with patch("graph_rag.llm.model_client.cost_tracker.record") as record:
            answer = client.generate(
                "What is in this source?",
                system_prompt="Use only the source.",
            )

        self.assertEqual(answer, "Grounded response")
        self.assertEqual(
            record.call_args.kwargs["usage"]["prompt_tokens"],
            8,
        )

    def test_embedding_client_preserves_provider_token_usage(self) -> None:
        from unittest.mock import patch
        from graph_rag.llm.model_client import CloudModelClient, model_request_context

        class FakeEmbeddings:
            chunk_size = 1000
            _invocation_params = {"model": "qwen/qwen3-embedding-8b"}

            def _ensure_sync_client_available(self):
                return None

            class Client:
                def create(self, input, **kwargs):
                    return {
                        "data": [
                            {"index": index, "embedding": [0.1, 0.2]}
                            for index, _ in enumerate(input)
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "total_tokens": 10,
                            "cost": 0.0009,
                        },
                    }

            client = Client()

        client = CloudModelClient(
            api_key="mock-key",
            embed_model="qwen/qwen3-embedding-8b",
            role="embedding",
        )
        client._get_embedding_model = lambda: FakeEmbeddings()
        with patch("graph_rag.llm.model_client.cost_tracker.record") as record:
            with model_request_context(run_id="ingestion:17", user_id=23):
                result = client.get_embeddings_batch(["source text"])

        self.assertEqual(result, [[0.1, 0.2]])
        self.assertEqual(record.call_args.kwargs["operation"], "embedding")
        self.assertEqual(record.call_args.kwargs["usage"]["prompt_tokens"], 10)
        self.assertEqual(record.call_args.kwargs["usage"]["completion_tokens"], 0)
        self.assertEqual(record.call_args.kwargs["run_id"], "ingestion:17")
        self.assertEqual(record.call_args.kwargs["user_id"], 23)

    def test_chat_configuration_failure_is_recorded(self) -> None:
        from unittest.mock import patch

        from graph_rag.llm.model_client import CloudModelClient, ModelConfigurationError

        client = CloudModelClient(
            api_key="",
            llm_model="mock-model",
            role="chat",
        )
        with patch("graph_rag.llm.model_client.cost_tracker.record") as record:
            with self.assertRaises(ModelConfigurationError):
                client.generate("question")

        self.assertEqual(record.call_args.kwargs["status"], "failed")
        self.assertIn("OPENROUTER_API_KEY", record.call_args.kwargs["error_message"])

    def test_langchain_usage_callback_records_usage_and_failure_context(self) -> None:
        import uuid
        from unittest.mock import patch

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult
        from graph_rag.llm.model_client import (
            ModelUsageCallbackHandler,
            model_request_context,
        )

        run_id = uuid.uuid4()
        with model_request_context(run_id="query-42", user_id=9):
            callback = ModelUsageCallbackHandler(
                "extraction",
                "qwen/qwen3.7-flash",
            )
            callback.on_chat_model_start({}, [[]], run_id=run_id)
            response = LLMResult(
                generations=[[
                    ChatGeneration(
                        message=AIMessage(
                            content="{}",
                            usage_metadata={
                                "input_tokens": 12,
                                "output_tokens": 4,
                                "total_tokens": 16,
                            },
                            response_metadata={
                                "finish_reason": "length",
                                "id": "mock-response-1",
                                "cost": 0.25,
                            },
                        )
                    )
                ]]
            )
            with patch("graph_rag.llm.model_client.cost_tracker.record") as record:
                callback.on_llm_end(response, run_id=run_id)

        call = record.call_args.kwargs
        self.assertEqual(call["status"], "success")
        self.assertEqual(call["usage"]["prompt_tokens"], 12)
        self.assertEqual(call["usage"]["completion_tokens"], 4)
        self.assertEqual(call["run_id"], "query-42")
        self.assertEqual(call["user_id"], 9)
        self.assertEqual(len(callback.response_records), 1)
        self.assertEqual(callback.response_records[0]["raw_model_output"], "{}")
        self.assertEqual(callback.response_records[0]["finish_reason"], "length")
        self.assertTrue(callback.response_records[0]["truncated"])
        self.assertEqual(callback.response_records[0]["provider_cost_usd"], 0.25)
        self.assertAlmostEqual(
            callback.response_records[0]["estimated_cost_usd"],
            0.00000088,
        )
        self.assertEqual(
            callback.response_records[0]["pricing_version"],
            "2026-11-26",
        )
        self.assertEqual(
            callback.response_records[0]["response_id"],
            "mock-response-1",
        )

        failed_run_id = uuid.uuid4()
        with model_request_context(run_id="query-43", user_id=10):
            callback = ModelUsageCallbackHandler("extraction", "mock-model")
            callback.on_chat_model_start({}, [[]], run_id=failed_run_id)
            with patch("graph_rag.llm.model_client.cost_tracker.record") as record:
                callback.on_llm_error(
                    RuntimeError("mock failure"),
                    run_id=failed_run_id,
                )

        self.assertEqual(record.call_args.kwargs["status"], "failed")
        self.assertEqual(record.call_args.kwargs["run_id"], "query-43")
        self.assertEqual(record.call_args.kwargs["error_message"], "mock failure")

    def test_evaluation_failures_are_unavailable_not_fabricated_scores(self) -> None:
        from unittest.mock import patch

        from graph_rag.evaluation.metrics import evaluate_all_metrics

        unavailable = (None, "Evaluation unavailable")
        with patch(
            "graph_rag.evaluation.metrics.evaluation_model_client.generate",
            side_effect=RuntimeError("mock provider failure"),
        ):
            result = evaluate_all_metrics("question", "answer", "context")

        self.assertEqual(result["faithfulness"], unavailable)
        self.assertEqual(result["answer_relevance"], unavailable)
        self.assertEqual(result["context_relevance"], unavailable)

        with patch(
            "graph_rag.evaluation.metrics.evaluation_model_client.generate",
            return_value='{"faithfulness":{"score":0.91,"reasoning":"supported"}}',
        ):
            partial = evaluate_all_metrics("question", "answer", "context")
        self.assertEqual(partial["faithfulness"], (0.91, "supported"))
        self.assertEqual(partial["answer_relevance"], unavailable)
        self.assertEqual(partial["context_relevance"], unavailable)

    def test_usage_summary_preserves_unknowns_and_failures(self) -> None:
        from graph_rag.database.db import init_db
        from graph_rag.llm.cost_tracker import CostTracker

        init_db()
        tracker = CostTracker()
        since_id = tracker.last_id()
        tracker.record(
            role="embedding",
            operation="embedding",
            model="mock-embedding",
            endpoint="/embeddings",
            usage={},
            request_id="unknown-usage",
        )
        tracker.record(
            role="chat",
            operation="generation",
            model="google/gemma-4-31b-it:free",
            endpoint="/chat/completions",
            usage={},
            request_id="failed-request",
            status="failed",
            error_message="mock failure",
        )

        summary = tracker.summary(since_id=since_id)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["unknown_cost_requests"], 1)
        self.assertEqual(summary["unknown_token_requests"], 2)
        recent = tracker.recent(limit=1)[0]
        self.assertIsNone(recent["total_tokens"])
        self.assertEqual(recent["cache_status"], "miss")

    def test_cost_tracker_calculates_cost_from_rate_card_not_provider_cost(self) -> None:
        from graph_rag.database.db import init_db
        from graph_rag.llm.cost_tracker import CostTracker
        from graph_rag.llm.pricing import calculate_cost_usd

        base_tier = calculate_cost_usd("qwen/qwen3.7-flash", 1000, 1000)
        high_tier = calculate_cost_usd("qwen/qwen3.7-flash", 32000, 1000)
        free_tier = calculate_cost_usd("google/gemma-4-31b-it:free", None, None)
        configured_free_models = (
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3.5-super-vl:free",
            "nvidia/nemotron-3.5-embed-8b:free",
        )
        configured_free_costs = [
            calculate_cost_usd(model, None, None)
            for model in configured_free_models
        ]
        unknown_model = calculate_cost_usd("mock-model", 100, 100)
        self.assertAlmostEqual(float(base_tier.cost_usd), 0.00016)
        self.assertAlmostEqual(float(high_tier.cost_usd), 0.0036)
        self.assertIn("prompt_tokens>=32000", high_tier.method)
        self.assertEqual(free_tier.cost_usd, 0)
        self.assertTrue(
            all(result.cost_usd == 0 for result in configured_free_costs)
        )
        self.assertIsNone(unknown_model.cost_usd)

        init_db()
        tracker = CostTracker()
        since_id = tracker.last_id()
        tracker.record(
            role="embedding",
            operation="embedding",
            model="qwen/qwen3-embedding-8b",
            endpoint="/embeddings",
            usage={
                "prompt_tokens": 10000,
                "completion_tokens": 0,
                "total_tokens": 10000,
                "cost": 0.0009,
            },
            request_id="priced-embedding",
        )
        tracker.record(
            role="embedding",
            operation="embedding",
            model="qwen/qwen3-embedding-8b",
            endpoint="/embeddings",
            usage={},
            request_id="missing-embedding-usage",
        )

        summary = tracker.summary(since_id=since_id)
        recent_by_request = {
            row["request_id"]: row for row in tracker.recent(limit=2)
        }
        recent = recent_by_request["priced-embedding"]
        self.assertAlmostEqual(summary["cost_usd"], 0.0001)
        self.assertEqual(summary["unknown_cost_requests"], 1)
        self.assertAlmostEqual(recent["cost_usd"], 0.0001)
        self.assertAlmostEqual(recent["provider_cost_usd"], 0.0009)
        self.assertEqual(
            recent["cost_method"],
            "local_rate_card:2026-11-26",
        )
        self.assertEqual(recent["input_rate_usd_per_million"], 0.01)
        self.assertEqual(recent["cache_status"], "miss")
        missing_usage = recent_by_request["missing-embedding-usage"]
        self.assertIsNone(missing_usage["cost_usd"])
        self.assertEqual(missing_usage["cost_method"], "missing_token_usage")

    def test_existing_model_usage_table_gets_additive_migration(self) -> None:
        import sqlite3

        from graph_rag.database.db import init_db, get_db_connection

        original_db_path = settings.SQLITE_DB_PATH
        migrated_path = self._temporary_root / "legacy.sqlite"
        try:
            object.__setattr__(settings, "SQLITE_DB_PATH", migrated_path)
            with sqlite3.connect(migrated_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE model_usage (
                        id INTEGER PRIMARY KEY,
                        role TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        model TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        cost_usd REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE graph_evidence (
                        id INTEGER PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        document_name TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        object_type TEXT NOT NULL,
                        object_id TEXT NOT NULL,
                        evidence_text TEXT NOT NULL,
                        page_number INTEGER,
                        sheet_name TEXT NOT NULL DEFAULT '',
                        section TEXT NOT NULL DEFAULT '',
                        source_offsets_json TEXT NOT NULL DEFAULT '[]',
                        bounding_boxes_json TEXT NOT NULL DEFAULT '[]',
                        confidence REAL NOT NULL,
                        extraction_model TEXT NOT NULL,
                        extraction_version TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO model_usage
                    (role, operation, model, endpoint, prompt_tokens,
                     completion_tokens, total_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "embedding",
                        "embedding",
                        "qwen/qwen3-embedding-8b",
                        "/embeddings",
                        10000,
                        0,
                        10000,
                        0.0009,
                    ),
                )
            init_db()
            with get_db_connection() as conn:
                columns = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(model_usage)"
                    ).fetchall()
                }
                evidence_columns = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(graph_evidence)"
                    ).fetchall()
                }
            self.assertTrue(
                {
                    "request_id",
                    "run_id",
                    "user_id",
                    "latency_ms",
                    "status",
                    "retry_count",
                    "cache_status",
                    "error_message",
                    "provider_cost_usd",
                    "input_rate_usd_per_million",
                    "output_rate_usd_per_million",
                    "cost_method",
                    "pricing_version",
                }.issubset(columns)
            )
            self.assertIn("mention_text", evidence_columns)
            with get_db_connection() as conn:
                row = conn.execute(
                """
                SELECT cost_usd, provider_cost_usd, cost_method
                FROM model_usage WHERE id = 1
                """
                ).fetchone()
            self.assertAlmostEqual(row["cost_usd"], 0.0001)
            self.assertAlmostEqual(row["provider_cost_usd"], 0.0009)
            self.assertEqual(row["cost_method"], "local_rate_card:2026-11-26")
        finally:
            object.__setattr__(settings, "SQLITE_DB_PATH", original_db_path)

    def test_replacing_document_chunks_removes_stale_chunk_ids(self) -> None:
        from unittest.mock import patch

        import chromadb
        import langchain_chroma
        from graph_rag.ingestion.chunker import TextChunk

        class FakeCollection:
            def __init__(self) -> None:
                self.ids = {"paper_p1_c0", "paper_p1_c1"}

            def upsert(self, ids, **kwargs) -> None:
                self.ids.update(ids)

            def get(self, **kwargs):
                return {"ids": list(self.ids)}

            def delete(self, ids) -> None:
                self.ids.difference_update(ids)

        class FakeClient:
            def __init__(self) -> None:
                self.collection = FakeCollection()

            def get_or_create_collection(self, **kwargs):
                return self.collection

        class FakeLangChainStore:
            def __init__(self, client, **kwargs) -> None:
                self.collection = client.collection

            def add_documents(self, documents, ids):
                self.collection.upsert(
                    ids=ids,
                    documents=[document.page_content for document in documents],
                    metadatas=[document.metadata for document in documents],
                )

        client = FakeClient()
        with (
            patch.object(chromadb, "PersistentClient", return_value=client),
            patch.object(langchain_chroma, "Chroma", FakeLangChainStore),
        ):
            from graph_rag.embeddings.vector_store import VectorStore

            store = VectorStore()
            chunks = [
                TextChunk("paper_p1_c0", "paper.pdf", "Paper", 0, 1, "same", 1),
                TextChunk("paper_p1_c2", "paper.pdf", "Paper", 2, 1, "new", 1),
            ]
            store.replace_document_chunks("paper.pdf", chunks)

        self.assertEqual(client.collection.ids, {"paper_p1_c0", "paper_p1_c2"})

    def test_open_entity_and_relation_types_require_grounded_evidence(self) -> None:
        from graph_rag.graph.entity_extractor import EntityExtractor

        result = EntityExtractor(client=object())._parse_json_response(
            """{
              "entities": [
                {"name": "Climate Event", "category": "Environmental Phenomenon",
                 "evidence": "Climate events influence crop yield.", "confidence": 0.95},
                {"name": "Crop Yield", "category": "Metric",
                 "evidence": "Climate events influence crop yield.", "confidence": 0.95}
              ],
              "mentions": [],
              "relations": [
                {"source": "Climate Event", "target": "Crop Yield",
                 "relation_type": "INFLUENCES",
                 "evidence": "Climate events influence crop yield.", "confidence": 0.95},
                {"source": "Climate Event", "target": "Crop Yield",
                 "relation_type": "UNSUPPORTED CLAIM", "evidence": "Unstated relationship.",
                 "confidence": 0.99}
              ]
            }""",
            source_text="Climate events influence crop yield.",
        )

        self.assertEqual(result.entities[0].category, "Environmental Phenomenon")
        self.assertEqual(len(result.relations), 1)
        self.assertEqual(result.relations[0].relation_type, "INFLUENCES")

    def test_arabic_evidence_matching_preserves_original_source_offsets(self) -> None:
        from graph_rag.graph.entity_extractor import EntityExtractor
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        source_text = "إِبْراهيم زار الـقاهرة."
        block = SourceBlock(
            source_ref="page:1",
            block_type="paragraph",
            text=source_text,
            page_number=1,
            start_offset=100,
            character_offsets=list(range(100, 100 + len(source_text))),
        )
        document = StructuredDocumentInput(
            document_id="arabic-report",
            title="Arabic report",
            document_type="report",
            language="ar",
            blocks=[block],
        )

        matched_block, span, reason = EntityExtractor._source_match_details(
            document,
            "page:1",
            "ابراهيم زار القاهرة",
        )

        self.assertIs(matched_block, block)
        self.assertEqual(reason, "")
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(source_text[start:end], "إِبْراهيم زار الـقاهرة")
        provenance = EntityExtractor._source_provenance(document, block, span)
        self.assertEqual(
            provenance["source_offsets"],
            [{"start": 100 + start, "end": 100 + end}],
        )

    def test_arabic_evidence_matching_rejects_repeated_matches_as_ambiguous(self) -> None:
        from graph_rag.graph.entity_extractor import EntityExtractor
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        document = StructuredDocumentInput(
            document_id="arabic-report",
            title="Arabic report",
            document_type="report",
            language="ar",
            blocks=[
                SourceBlock(
                    source_ref="page:1",
                    block_type="paragraph",
                    text="زار إبراهيم القاهرة. زار ابراهيم القاهرة.",
                )
            ],
        )

        block, span, reason = EntityExtractor._source_match_details(
            document,
            "",
            "زار إبراهيم القاهرة",
        )

        self.assertIsNone(block)
        self.assertIsNone(span)
        self.assertEqual(reason, "ambiguous_evidence")

    def test_arabic_evidence_matching_tolerates_small_ocr_error(self) -> None:
        from graph_rag.graph.entity_extractor import EntityExtractor
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        source_text = "يشهد الاقتصاد العالمي تحولات كبيرة ومتسارعة."
        block = SourceBlock(
            source_ref="page:1",
            block_type="paragraph",
            text=source_text,
            page_number=1,
        )
        document = StructuredDocumentInput(
            document_id="arabic-report",
            title="آفاق اقتصادية",
            document_type="report",
            language="ar",
            blocks=[block],
        )

        matched_block, span, reason = EntityExtractor._source_match_details(
            document,
            "page:1",
            "يشهد الاقتصاد العالني تحولات كبيرة ومتسارعة",
        )

        self.assertIs(matched_block, block)
        self.assertEqual(reason, "")
        self.assertIsNotNone(span)

    def test_arabic_node_name_prefers_source_language_and_keeps_alias(self) -> None:
        from graph_rag.graph.entity_extractor import EntityExtractor

        name, aliases = EntityExtractor._canonical_node_name(
            "World_Bank",
            {"name": "World Bank"},
            ["البنك الدولي", "WB"],
            "mixed",
        )

        self.assertEqual(name, "البنك الدولي")
        self.assertIn("World Bank", aliases)
        self.assertIn("World_Bank", aliases)

    def test_long_document_is_split_by_character_budget(self) -> None:
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
            split_document_input,
        )

        document = StructuredDocumentInput(
            document_id="long-report",
            title="Long report",
            document_type="report",
            language="ar",
            blocks=[
                SourceBlock(
                    source_ref=f"page:{index}",
                    block_type="paragraph",
                    text=("اقتصاد عالمي وتحول رقمي. " * 500),
                    page_number=index,
                )
                for index in range(1, 7)
            ],
        )

        segments = split_document_input(document, 40000, len)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment.render()) <= 40000 for segment in segments))

    def test_graph_extraction_stores_the_original_arabic_evidence_quote(self) -> None:
        from types import SimpleNamespace
        from graph_rag.graph.entity_extractor import EntityExtractor
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        source_text = "إِبْراهيم زار الـقاهرة."
        entity = SimpleNamespace(
            id="إبراهيم",
            type="Person",
            properties={
                "evidence": "ابراهيم زار القاهرة",
                "source_ref": "page:1",
                "confidence": 0.96,
            },
        )
        graph_document = SimpleNamespace(nodes=[entity], relationships=[])
        bound_options = {}

        class BindableChatModel:
            def bind(self, **kwargs):
                bound_options.update(kwargs)
                return self

        document = StructuredDocumentInput(
            document_id="arabic-report",
            title="Arabic report",
            document_type="report",
            language="ar",
            blocks=[
                SourceBlock(
                    source_ref="page:1",
                    block_type="paragraph",
                    text=source_text,
                    page_number=1,
                    character_offsets=list(range(len(source_text))),
                )
            ],
        )

        class FakeTransformer:
            def __init__(self, **kwargs):
                pass

            def process_response(self, source, config):
                return graph_document

        with (
            patch(
                "graph_rag.graph.entity_extractor.LLMGraphTransformer",
                FakeTransformer,
            ),
            patch(
                "graph_rag.graph.entity_extractor.JsonModeChatModel",
                side_effect=lambda model: model,
            ),
            patch(
                "graph_rag.graph.entity_extractor.ModelUsageCallbackHandler",
                return_value=SimpleNamespace(response_records=[]),
            ),
        ):
            result = EntityExtractor(
                client=SimpleNamespace(as_chat_model=lambda: BindableChatModel())
            ).extract_document(document)

        self.assertEqual(
            bound_options["max_tokens"],
            settings.EXTRACTION_OUTPUT_TOKENS,
        )
        self.assertEqual(
            bound_options["reasoning"],
            {"enabled": settings.EXTRACTION_REASONING_ENABLED},
        )
        self.assertEqual(result.entities[0].evidence, source_text[:-1])
        self.assertEqual(
            result.entities[0].provenance["source_offsets"],
            [{"start": 0, "end": len(source_text) - 1}],
        )

    def test_graph_transformer_prompt_escapes_literal_json_braces(self) -> None:
        from graph_rag.graph.entity_extractor import GRAPH_TRANSFORMER_PROMPT

        self.assertEqual(GRAPH_TRANSFORMER_PROMPT.input_variables, ["input"])

        rendered = GRAPH_TRANSFORMER_PROMPT.format(
            input='{"nodes": [], "relationships": []}'
        )

        self.assertIn(
            'Properties are arrays of {"key": "...", "value": "..."} strings.',
            rendered,
        )
        self.assertIn('{"nodes": [], "relationships": []}', rendered)

    def test_document_extraction_uses_open_graph_transformer_and_exact_evidence(self) -> None:
        from types import SimpleNamespace
        from graph_rag.graph.entity_extractor import EntityExtractor
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        source_text = "Climate Event influences Crop Yield."
        source_ref = "page:1"
        climate = SimpleNamespace(
            id="Climate Event",
            type="Environmental Phenomenon",
            properties={
                "evidence": source_text,
                "source_ref": source_ref,
                "confidence": "0.96",
                "aliases": '["Climate Event"]',
                "attributes": "{}",
            },
        )
        crop = SimpleNamespace(
            id="Crop Yield",
            type="Agricultural Metric",
            properties={
                "evidence": source_text,
                "source_ref": source_ref,
                "confidence": "0.96",
                "aliases": "[]",
                "attributes": "{}",
            },
        )
        sentence = SimpleNamespace(
            id="This is an unsupported full sentence.",
            type="Finding",
            properties={
                "evidence": source_text,
                "source_ref": source_ref,
                "confidence": "0.99",
            },
        )
        graph_document = SimpleNamespace(
            nodes=[climate, crop, sentence],
            relationships=[
                SimpleNamespace(
                    source=climate,
                    target=crop,
                    type="INFLUENCES",
                    properties={
                        "evidence": source_text,
                        "source_ref": source_ref,
                        "confidence": "0.95",
                        "description": "Climate affects crop yield.",
                    },
                )
            ],
        )
        transformer_options = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                transformer_options.update(kwargs)

            def process_response(self, document, config):
                return graph_document

        source = StructuredDocumentInput(
            document_id="climate-report",
            title="Climate report",
            document_type="report",
            language="en",
            blocks=[
                SourceBlock(
                    source_ref=source_ref,
                    block_type="paragraph",
                    text=source_text,
                    page_number=1,
                    section="Findings",
                    character_offsets=list(range(len(source_text))),
                )
            ],
        )

        with (
            patch(
                "graph_rag.graph.entity_extractor.LLMGraphTransformer",
                FakeTransformer,
            ),
            patch(
                "graph_rag.graph.entity_extractor.JsonModeChatModel",
                side_effect=lambda model: model,
            ),
            patch(
                "graph_rag.graph.entity_extractor.ModelUsageCallbackHandler",
                return_value=SimpleNamespace(response_records=[]),
            ),
        ):
            result = EntityExtractor(
                client=SimpleNamespace(as_chat_model=lambda: object())
            ).extract_document(source)

        self.assertEqual(transformer_options["allowed_nodes"], [])
        self.assertEqual(transformer_options["allowed_relationships"], [])
        self.assertEqual(
            transformer_options["node_properties"],
            [
                "name",
                "document_type",
                "evidence",
                "source_ref",
                "confidence",
                "aliases",
                "attributes",
                "description",
            ],
        )
        self.assertEqual(
            transformer_options["relationship_properties"],
            ["evidence", "source_ref", "confidence", "description"],
        )
        self.assertEqual(
            [entity.category for entity in result.entities],
            ["Environmental Phenomenon", "Agricultural Metric"],
        )
        self.assertEqual(len(result.relations), 1)
        self.assertEqual(result.relations[0].relation_type, "INFLUENCES")
        self.assertTrue(all(entity.provenance["page_number"] == 1 for entity in result.entities))
        rejected_node = next(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic["diagnostic_type"] == "rejected_node"
        )
        self.assertIn("invalid_entity_name", rejected_node["rejection_reasons"])

    def test_document_extraction_error_retains_raw_model_response(self) -> None:
        from types import SimpleNamespace
        from graph_rag.graph.entity_extractor import (
            EntityExtractor,
            GraphExtractionError,
        )
        from graph_rag.ingestion.document_structure import (
            SourceBlock,
            StructuredDocumentInput,
        )

        source = StructuredDocumentInput(
            document_id="report",
            title="Report",
            document_type="report",
            language="ar",
            blocks=[
                SourceBlock(
                    source_ref="page:1",
                    block_type="paragraph",
                    text="النص العربي.",
                    page_number=1,
                )
            ],
        )
        response_record = {
            "raw_model_output": '{"nodes":',
            "finish_reason": "length",
            "truncated": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

        class FakeTransformer:
            def __init__(self, **kwargs):
                pass

            def process_response(self, document, config):
                raise ValueError("mock truncated JSON")

        with (
            patch(
                "graph_rag.graph.entity_extractor.LLMGraphTransformer",
                FakeTransformer,
            ),
            patch(
                "graph_rag.graph.entity_extractor.JsonModeChatModel",
                side_effect=lambda model: model,
            ),
            patch(
                "graph_rag.graph.entity_extractor.ModelUsageCallbackHandler",
                return_value=SimpleNamespace(response_records=[response_record]),
            ),
        ):
            with self.assertRaises(GraphExtractionError) as error:
                EntityExtractor(
                    client=SimpleNamespace(as_chat_model=lambda: object())
                ).extract_document(source)

        self.assertEqual(
            error.exception.model_responses[0]["raw_model_output"],
            '{"nodes":',
        )
        self.assertEqual(error.exception.model_responses[0]["language"], "ar")
        self.assertEqual(
            error.exception.model_responses[0]["source_refs"],
            ["page:1"],
        )

    def test_graph_extraction_uses_json_mode_and_parses_ai_message(self) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        from graph_rag.graph.entity_extractor import EntityExtractor

        class FakeChatModel:
            def bind(self, **kwargs):
                if "response_format" in kwargs:
                    assert kwargs["response_format"] == {"type": "json_object"}
                    return RunnableLambda(respond)
                return self

        def respond(prompt_value):
            prompt_text = prompt_value.to_string()
            assert "Nimble Atlas validates on Quantum Loom." in prompt_text
            return AIMessage(
                content=json.dumps(
                    {
                        "entities": [
                            {
                                "name": "Nimble Atlas",
                                "category": "Computational Framework",
                                "evidence": "Nimble Atlas validates on Quantum Loom.",
                                "confidence": 0.95,
                            },
                            {
                                "name": "Quantum Loom",
                                "category": "Experimental Dataset",
                                "evidence": "Nimble Atlas validates on Quantum Loom.",
                                "confidence": 0.95,
                            },
                        ],
                        "relations": [
                            {
                                "source": "Nimble Atlas",
                                "target": "Quantum Loom",
                                "relation_type": "VALIDATES_ON",
                                "evidence": (
                                    "Nimble Atlas validates on Quantum Loom."
                                ),
                                "confidence": 0.95,
                            }
                        ],
                    }
                )
            )

        class FakeClient:
            def as_chat_model(self):
                return FakeChatModel()

        result = EntityExtractor(client=FakeClient()).extract(
            "Nimble Atlas validates on Quantum Loom.",
            document_title="A Study",
            chunk_id="study_p1_c0",
        )

        self.assertEqual(result.entities[0].category, "Computational Framework")
        self.assertEqual(result.relations[0].relation_type, "VALIDATES_ON")
        self.assertEqual(len(result.relations), 1)

    def test_graph_extraction_accepts_raw_structured_responses(self) -> None:
        from langchain_core.messages import AIMessage
        from graph_rag.graph.entity_extractor import EntityExtractor

        payload = {
            "entities": [
                {
                    "name": "Nimble Atlas",
                    "category": "Computational Framework",
                    "description": "",
                    "aliases": [],
                    "attributes": {},
                    "evidence": "Nimble Atlas validates on Quantum Loom.",
                    "confidence": 0.95,
                }
            ],
            "mentions": [],
            "relations": [],
            "topics": [],
        }
        raw_json = json.dumps(payload)
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "GraphExtractionOutput",
                    "args": payload,
                    "id": "call_graph_extraction",
                    "type": "tool_call",
                }
            ],
        )
        extractor = EntityExtractor(client=object())

        for output in (raw_json, AIMessage(content=raw_json), tool_call_message):
            with self.subTest(output_type=type(output).__name__):
                result = extractor._parse_structured_output(
                    output,
                    chunk_id="study:paper-level",
                    source_text="Nimble Atlas validates on Quantum Loom.",
                )
                self.assertEqual(result.entities[0].name, "Nimble Atlas")

    def test_graph_writer_stores_open_relation_as_data(self) -> None:
        from graph_rag.graph.entity_extractor import (
            ExtractedEntity,
            ExtractedRelation,
            GraphExtractionResult,
        )
        from graph_rag.graph.graph_builder import GraphBuilder

        class FakeResult:
            def consume(self):
                return self

            def single(self):
                return None

        class FakeSession:
            def __init__(self):
                self.queries = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def run(self, query, **params):
                self.queries.append((query, params))
                return FakeResult()

        class FakeDriver:
            def __init__(self):
                self.session_instance = FakeSession()

            def session(self):
                return self.session_instance

        builder = GraphBuilder()
        driver = FakeDriver()
        builder._driver = driver
        result = GraphExtractionResult(
            entities=[
                ExtractedEntity("Rainfall", "Weather Event", entity_id="a"),
                ExtractedEntity("Harvest", "Agricultural Outcome", entity_id="b"),
            ],
            relations=[
                ExtractedRelation(
                    source="Rainfall",
                    target="Harvest",
                    relation_type="INFLUENCES",
                    source_entity_id="a",
                    target_entity_id="b",
                    evidence="Rainfall influences harvest.",
                )
            ],
        )

        self.assertTrue(builder.add_entities_and_relations("sample.parquet", result))
        relation_query = next(
            query
            for query, _ in driver.session_instance.queries
            if "MERGE (s)-[r:RELATION" in query
        )
        self.assertIn("relation_type: $relation_type", relation_query)


if __name__ == "__main__":
    unittest.main()
