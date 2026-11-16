"""
Generate DOCX version of evaluation report.
"""

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import json

# Load settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_rag.config import settings

# Paths
EVAL_DIR = Path("eval_results")
REPORT_MD = EVAL_DIR / "EVALUATION_REPORT.md"
SUMMARY_FILE = EVAL_DIR / "summary.json"
CHART1 = EVAL_DIR / "metrics_per_paper.png"
CHART2 = EVAL_DIR / "precision_per_relation.png"

# Load summary
with open(SUMMARY_FILE) as f:
    summary = json.load(f)

# Create document
doc = Document()

# Title
title = doc.add_heading('Knowledge Graph Evaluation Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Metadata
doc.add_heading('Evaluation Metadata', level=2)
p = doc.add_paragraph()
p.add_run(f"Evaluation Date: {summary['evaluation_date']}\n")
p.add_run(f"Dataset ID: {summary['dataset_id']}\n")
p.add_run(f"Total Papers: {summary['total_papers']}\n")
p.add_run(f"Total Entities: {summary['total_entities']}\n")
p.add_run(f"Total Triples Evaluated: {summary['total_triples']}")

# Executive Summary
doc.add_heading('Executive Summary', level=2)
doc.add_paragraph(
    f"The knowledge graph evaluation reveals significant quality issues that require immediate attention. "
    f"Overall relation precision is {summary['overall_relation_precision']}%, far below the 80% threshold, "
    f"indicating that the majority of extracted relationships are not supported by the source text. "
    f"Entity grounding is relatively strong at 81-84% across papers, meeting the 90% threshold only partially. "
    f"However, duplicate rates range from 7-18%, exceeding the 5% target in two of three papers. "
    f"Orphan rates are 8-14%, slightly above the 10% threshold in two papers. "
    f"Source support is perfect at 100%, but this does not translate to accurate relationships. "
    f"The most problematic relationship types are REPORTS (1.54% precision), EVALUATES_ON (4.48%), "
    f"and PART_OF_TOPIC (13.33%), while AFFILIATED_WITH performs best at 78.95%. "
    f"Cross-paper entity sharing is minimal (0-1% Jaccard overlap), with no entities shared across all three papers."
)

p = doc.add_paragraph()
p.add_run("Verdict: Needs Work").bold = True

# Method
doc.add_heading('Method', level=2)

doc.add_heading('Graph Schema', level=3)
doc.add_paragraph(
    "The evaluation targets an evidence-first Neo4j knowledge graph with the following structure:"
)
doc.add_paragraph("Entity nodes: uid, name, aliases, entity_id, category, description, attributes_json, dataset_id", style='List Bullet')
doc.add_paragraph("Paper nodes: uid, paper_id, title, source, dataset_id", style='List Bullet')
doc.add_paragraph("Chunk nodes: Text chunks with page numbers and text previews", style='List Bullet')
doc.add_paragraph("Mention nodes: Link chunks to entities with evidence and confidence", style='List Bullet')
doc.add_paragraph("Semantic relationships: RELATED_TO, USES_METHOD, EVALUATES_ON, REPORTS, PROPOSES, AUTHORED, AFFILIATED_WITH, PART_OF_TOPIC", style='List Bullet')
doc.add_paragraph("Structural relationships (excluded): HAS_PAPER, HAS_CHUNK, MENTIONS, CONTAINS_MENTION, REFERS_TO", style='List Bullet')

doc.add_heading('Models Used', level=3)
doc.add_paragraph(f"Extraction model: {settings.EXTRACTION_MODEL} (configured in .env)")
doc.add_paragraph(f"Judge model: {settings.EVALUATION_MODEL} (temperature=0.0, JSON-only output)")
doc.add_paragraph("The judge is a stronger model than the extraction model, configured to be strict and conservative.")

doc.add_heading('Sample Size', level=3)
doc.add_paragraph(f"Total triples evaluated: {summary['total_triples']} (all triples, no sampling)")
doc.add_paragraph(f"Papers evaluated: {summary['total_papers']}")
doc.add_paragraph("Random seed: 42 (for reproducibility)")

# Per-Paper Results
doc.add_heading('Per-Paper Results', level=2)

# Table
table = doc.add_table(rows=1, cols=9)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Paper'
hdr_cells[1].text = 'Entities'
hdr_cells[2].text = 'Triples'
hdr_cells[3].text = 'Entity Grounding %'
hdr_cells[4].text = 'Duplicate Rate %'
hdr_cells[5].text = 'Orphan Rate %'
hdr_cells[6].text = 'Relation Precision %'
hdr_cells[7].text = 'Source Support %'
hdr_cells[8].text = 'Entity Precision %'

for paper_id, data in summary["per_paper_results"].items():
    row_cells = table.add_row().cells
    row_cells[0].text = paper_id[:40] + "..."
    row_cells[1].text = str(data['entity_count'])
    row_cells[2].text = str(data['triple_count'])
    row_cells[3].text = str(data['entity_grounding_pct'])
    row_cells[4].text = str(data['duplicate_rate_pct'])
    row_cells[5].text = str(data['orphan_rate_pct'])
    row_cells[6].text = str(data['relation_precision_pct'])
    row_cells[7].text = str(data['source_support_pct'])
    row_cells[8].text = str(data['entity_precision_pct'])

# Add chart
doc.add_heading('Quality Metrics per Paper', level=3)
doc.add_picture(str(CHART1), width=Inches(6))

# Relationship Type Analysis
doc.add_heading('Relationship-Type Analysis', level=2)

# Table
table = doc.add_table(rows=1, cols=4)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Relationship Type'
hdr_cells[1].text = 'Total Triples'
hdr_cells[2].text = 'Supported'
hdr_cells[3].text = 'Precision %'

for rel_type, precision in summary["relation_type_precision"].items():
    row_cells = table.add_row().cells
    row_cells[0].text = rel_type
    row_cells[1].text = str(summary["relation_type_counts"][rel_type]["total"])
    row_cells[2].text = str(summary["relation_type_counts"][rel_type]["supported"])
    row_cells[3].text = str(precision)

# Add chart
doc.add_heading('Precision per Relationship Type', level=3)
doc.add_picture(str(CHART2), width=Inches(6))

doc.add_heading('Interpretation', level=3)
doc.add_paragraph("Reliable relationship types (precision >= 80%):")
doc.add_paragraph("AFFILIATED_WITH (78.95%): Nearly meets threshold, likely accurate for author affiliations", style='List Bullet')
doc.add_paragraph()
doc.add_paragraph("Problematic relationship types (precision < 50%):")
doc.add_paragraph("REPORTS (1.54%): Extremely poor; the model incorrectly infers reporting relationships", style='List Bullet')
doc.add_paragraph("EVALUATES_ON (4.48%): Very poor; evaluation claims are rarely supported by text", style='List Bullet')
doc.add_paragraph("PART_OF_TOPIC (13.33%): Poor; topic categorization is unreliable", style='List Bullet')
doc.add_paragraph("USES_METHOD (18.62%): Poor; method usage claims are frequently hallucinated", style='List Bullet')
doc.add_paragraph("RELATED_TO (21.53%): Poor; generic relationships are too vague and often unsupported", style='List Bullet')
doc.add_paragraph("PROPOSES (25.0%): Poor; proposal claims are rarely explicit in text", style='List Bullet')
doc.add_paragraph("AUTHORED (26.32%): Poor; authorship extraction is unreliable", style='List Bullet')

# Cross-Paper Analysis
doc.add_heading('Cross-Paper Analysis', level=2)

doc.add_heading('Pairwise Jaccard Overlap', level=3)
table = doc.add_table(rows=1, cols=4)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Paper Pair'
hdr_cells[1].text = 'Jaccard'
hdr_cells[2].text = 'Intersection'
hdr_cells[3].text = 'Union'

for key, data in summary["cross_paper_analysis"]["overlap_matrix"].items():
    row_cells = table.add_row().cells
    row_cells[0].text = f"{data['paper1'][:30]}... / {data['paper2'][:30]}..."
    row_cells[1].text = str(data['jaccard'])
    row_cells[2].text = str(data['intersection'])
    row_cells[3].text = str(data['union'])

doc.add_heading('Top Shared Entities', level=3)
table = doc.add_table(rows=1, cols=4)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Normalized Name'
hdr_cells[1].text = 'Original Name'
hdr_cells[2].text = 'Paper Count'
hdr_cells[3].text = 'Papers'

for se in summary["cross_paper_analysis"]["shared_entities"][:15]:
    row_cells = table.add_row().cells
    row_cells[0].text = se['normalized_name']
    row_cells[1].text = se['original_name']
    row_cells[2].text = str(se['paper_count'])
    row_cells[3].text = ', '.join([p[:30]+'...' for p in se['papers']])

doc.add_paragraph(
    f"Cross-paper entity sharing is minimal. Jaccard overlap ranges from 0% to 1.04%, "
    f"indicating that the three papers cover distinct domains with little conceptual overlap. "
    f"There are {len(summary['cross_paper_analysis']['entities_in_all_papers'])} entities shared across all three papers. "
    f"The shared entities are generic ML concepts rather than domain-specific entities."
)

# Thresholds
doc.add_heading('Thresholds', level=2)
table = doc.add_table(rows=1, cols=5)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Metric'
hdr_cells[1].text = 'Target'
hdr_cells[2].text = 'Actual (Avg)'
hdr_cells[3].text = 'Status'

avg_grounding = round(sum(d['entity_grounding_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)
avg_duplicate = round(sum(d['duplicate_rate_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)
avg_orphan = round(sum(d['orphan_rate_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)

metrics_data = [
    ("Source Support", ">= 85%", "100%", "Pass"),
    ("Relation Precision", ">= 80%", f"{summary['overall_relation_precision']}%", "Fail"),
    ("Entity Grounding", ">= 90%", f"{avg_grounding}%", "Fail"),
    ("Duplicate Rate", "<= 5%", f"{avg_duplicate}%", "Fail"),
    ("Orphan Rate", "<= 10%", f"{avg_orphan}%", "Fail"),
]

for metric, target, actual, status in metrics_data:
    row_cells = table.add_row().cells
    row_cells[0].text = metric
    row_cells[1].text = target
    row_cells[2].text = actual
    row_cells[3].text = status

# Save
output_path = EVAL_DIR / "EVALUATION_REPORT.docx"
doc.save(output_path)
print(f"DOCX report saved: {output_path}")
