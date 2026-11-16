"""
Generate DOCX version of evaluation report (v2 with audit).
"""

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import json
import csv

# Load settings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_rag.config import settings

# Paths
EVAL_DIR = Path("eval_results")
SUMMARY_FILE = EVAL_DIR / "summary.json"
AUDIT_FILE = EVAL_DIR / "audit.csv"
ERRORS_FILE = EVAL_DIR / "errors_to_review.csv"
CHART1 = EVAL_DIR / "metrics_per_paper.png"
CHART2 = EVAL_DIR / "precision_per_relation.png"

# Load summary
with open(SUMMARY_FILE) as f:
    summary = json.load(f)

# Load audit results
audit_results = []
audit_agreement = 0
audit_kappa = 0.0
audit_precision = 0.0
audit_sample_size = 0

if AUDIT_FILE.exists():
    with open(AUDIT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        audit_results = list(reader)
    
    if audit_results:
        audit_sample_size = len(audit_results)
        agree_count = sum(1 for a in audit_results if a["agree"] == "AGREE")
        audit_agreement = round(agree_count / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0
        
        po = agree_count / audit_sample_size
        pe = 0.5
        if pe < 1:
            audit_kappa = round((po - pe) / (1 - pe), 2)
        
        supported_audit = sum(1 for a in audit_results if a["audit_verdict"] in ["explicit", "implied"])
        audit_precision = round(supported_audit / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0

# Load errors
errors = []
with open(ERRORS_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    errors = list(reader)

# Compute overall metrics
overall_rel_precision = summary["overall_metrics"]["relation_precision_pct"]
# Find paper with data
paper_with_data = None
for paper_id, data in summary["per_paper_results"].items():
    if data.get('triple_count', 0) > 0:
        paper_with_data = data
        break

if paper_with_data:
    overall_entity_grounding = paper_with_data.get("entity_grounding_pct", 0)
    overall_duplicate_rate = paper_with_data.get("duplicate_rate_pct", 0)
    overall_orphan_rate = paper_with_data.get("orphan_rate_pct", 0)
else:
    overall_entity_grounding = 0
    overall_duplicate_rate = 0
    overall_orphan_rate = 0

# Determine verdict
if audit_sample_size > 0:
    final_precision = audit_precision
    verdict_basis = "audit"
else:
    final_precision = overall_rel_precision
    verdict_basis = "judge"

verdict = "PASS"
reasons = []

if final_precision < 80:
    verdict = "FAIL"
    reasons.append(f"Relation precision ({final_precision}%) below 80%")
if overall_entity_grounding < 90:
    verdict = "FAIL"
    reasons.append(f"Entity grounding ({overall_entity_grounding}%) below 90%")
if overall_duplicate_rate > 5:
    verdict = "FAIL"
    reasons.append(f"Duplicate rate ({overall_duplicate_rate}%) exceeds 5%")
if overall_orphan_rate > 10:
    verdict = "FAIL"
    reasons.append(f"Orphan rate ({overall_orphan_rate}%) exceeds 10%")

verdict_text = f"PASS" if verdict == "PASS" else f"FAIL - {'; '.join(reasons)}"

# Create document
doc = Document()

# Title
title = doc.add_heading('Knowledge Graph Evaluation Report (v2)', 0)
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
    f"The knowledge graph evaluation was conducted using a three-model approach with independent audit. "
    f"The judge model evaluated all {summary['total_triples']} triples, achieving {overall_rel_precision}% relation precision. "
    f"An independent audit sampled {audit_sample_size} triples, achieving {audit_precision}% audit precision with "
    f"{audit_agreement}% agreement between judge and auditor (Cohen's κ = {audit_kappa})."
)

p = doc.add_paragraph()
p.add_run(f"Verdict: {verdict_text}").bold = True

# Method
doc.add_heading('Method', level=2)

doc.add_heading('Graph Schema', level=3)
doc.add_paragraph("The evaluation targets an evidence-first Neo4j knowledge graph with the following structure:")
doc.add_paragraph("Entity nodes: uid, name, aliases, entity_id, category, description, attributes_json, dataset_id", style='List Bullet')
doc.add_paragraph("Paper nodes: uid, paper_id, title, source, dataset_id", style='List Bullet')
doc.add_paragraph("Chunk nodes: Text chunks with text previews", style='List Bullet')
doc.add_paragraph("Mention nodes: Link chunks to entities with evidence and confidence", style='List Bullet')
doc.add_paragraph("Semantic relationships: RELATED_TO, USES_METHOD, EVALUATES_ON, REPORTS, PROPOSES, AUTHORED, AFFILIATED_WITH, PART_OF_TOPIC", style='List Bullet')
doc.add_paragraph("Structural relationships (excluded): HAS_PAPER, HAS_CHUNK, MENTIONS, CONTAINS_MENTION, REFERS_TO", style='List Bullet')

doc.add_heading('Models Used', level=3)
doc.add_paragraph(f"Extraction model: {summary.get('extractor_model', settings.EXTRACTION_MODEL)}")
doc.add_paragraph(f"Judge model: {summary.get('judge_model', settings.EVALUATION_MODEL)} (temperature=0.0, JSON-only output)")
doc.add_paragraph(f"Audit model: {summary.get('audit_model', 'N/A')} (independent verification with full paper text)")
doc.add_paragraph("The judge is a stronger model than the extraction model. The auditor is a third, distinct model that independently verifies triples against the full paper text.")

doc.add_heading('Sample Size', level=3)
doc.add_paragraph(f"Total triples evaluated: {summary['total_triples']} (all triples, no sampling)")
doc.add_paragraph(f"Audit sample size: {audit_sample_size} (stratified by paper and relation type)")
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
    row_cells[1].text = str(data.get('entity_count', 0))
    row_cells[2].text = str(data.get('triple_count', 0))
    row_cells[3].text = str(data.get('entity_grounding_pct', 0))
    row_cells[4].text = str(data.get('duplicate_rate_pct', 0))
    row_cells[5].text = str(data.get('orphan_rate_pct', 0))
    row_cells[6].text = str(data.get('relation_precision_pct', 0))
    row_cells[7].text = str(data.get('source_support_pct', 0))
    row_cells[8].text = str(data.get('entity_precision_pct', 0))

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

rel_types = list(summary["relation_type_precision"].keys())
reliable = [rt for rt in rel_types if summary["relation_type_precision"][rt] >= 80]
problematic = [rt for rt in rel_types if summary["relation_type_precision"][rt] < 50]

if reliable:
    doc.add_paragraph("Reliable relationship types (precision >= 80%):")
    for rt in reliable:
        doc.add_paragraph(f"{rt} ({summary['relation_type_precision'][rt]}%)", style='List Bullet')
    doc.add_paragraph()

if problematic:
    doc.add_paragraph("Problematic relationship types (precision < 50%):")
    for rt in problematic:
        doc.add_paragraph(f"{rt} ({summary['relation_type_precision'][rt]}%)", style='List Bullet')

# Audit Results
if audit_sample_size > 0:
    doc.add_heading('Independent Audit Results', level=2)
    doc.add_paragraph(
        f"The independent audit sampled {audit_sample_size} triples stratified by paper and relation type. "
        f"The auditor had access to the full paper text and was instructed to verify each triple against the source."
    )
    doc.add_paragraph(f"Audit precision: {audit_precision}% (percentage of audit verdicts that are explicit/implied)")
    doc.add_paragraph(f"Judge-auditor agreement: {audit_agreement}% ({sum(1 for a in audit_results if a['agree'] == 'AGREE')}/{audit_sample_size} samples)")
    doc.add_paragraph(f"Cohen's kappa: {audit_kappa}")
    
    quote_verified = sum(1 for a in audit_results if a["quote_verified"] == "True")
    quote_verification_rate = round(quote_verified / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0
    doc.add_paragraph(f"Quote verification rate: {quote_verification_rate}% (percentage of audit quotes found in paper text)")
    
    if summary.get('audit_model') == summary.get('judge_model'):
        doc.add_paragraph().add_run("Note: The audit model is the same as the judge model, which compromises independence.").italic = True
    
    # Verdict counts table
    doc.add_heading('Audit Sample Summary', level=3)
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Judge Verdict'
    hdr_cells[1].text = 'Audit Verdict'
    hdr_cells[2].text = 'Count'
    
    verdict_counts = {}
    for a in audit_results:
        key = f"{a['judge_verdict']} -> {a['audit_verdict']}"
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    
    for key, count in sorted(verdict_counts.items()):
        row_cells = table.add_row().cells
        parts = key.split(" -> ")
        row_cells[0].text = parts[0]
        row_cells[1].text = parts[1]
        row_cells[2].text = str(count)
    
    # Top disagreements
    disagreements = [a for a in audit_results if a["agree"] == "DISAGREE"]
    if disagreements:
        doc.add_heading('Top Judge-Auditor Disagreements', level=3)
        table = doc.add_table(rows=1, cols=7)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Paper'
        hdr_cells[1].text = 'Head'
        hdr_cells[2].text = 'Relation'
        hdr_cells[3].text = 'Tail'
        hdr_cells[4].text = 'Judge'
        hdr_cells[5].text = 'Auditor'
        hdr_cells[6].text = 'Quote'
        
        for d in disagreements[:5]:
            row_cells = table.add_row().cells
            row_cells[0].text = d['paper_id'][:30] + "..."
            row_cells[1].text = d['head'][:20] + "..."
            row_cells[2].text = d['relation']
            row_cells[3].text = d['tail'][:20] + "..."
            row_cells[4].text = d['judge_verdict']
            row_cells[5].text = d['audit_verdict']
            row_cells[6].text = d['quote'][:50] + "..." if len(d['quote']) > 50 else d['quote']

# Error Analysis
doc.add_heading('Error Analysis', level=2)
doc.add_paragraph(f"The {len(errors)} unsupported triples were analyzed for root causes:")

# Error categories
error_categories = {
    "hallucinated_entity": 0,
    "inference_not_stated": 0,
    "generic_vague_entity": 0,
    "missing_evidence": 0,
    "other": 0
}

for error in errors:
    reason = error["judge_reason"].lower()
    if "does not mention" in reason or "not mention" in reason:
        error_categories["hallucinated_entity"] += 1
    elif "only mentions" in reason or "does not explicitly state" in reason or "not explicitly" in reason:
        error_categories["inference_not_stated"] += 1
    elif "vague" in reason or "ambiguous" in reason or "generic" in reason:
        error_categories["generic_vague_entity"] += 1
    elif "no evidence" in reason or "insufficient evidence" in reason:
        error_categories["missing_evidence"] += 1
    else:
        error_categories["other"] += 1

# Error category table
doc.add_heading('Error Category Summary', level=3)
table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Category'
hdr_cells[1].text = 'Count'

for cat, count in error_categories.items():
    row_cells = table.add_row().cells
    row_cells[0].text = cat
    row_cells[1].text = str(count)

# Thresholds
doc.add_heading('Thresholds', level=2)
table = doc.add_table(rows=1, cols=5)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Metric'
hdr_cells[1].text = 'Target'
hdr_cells[2].text = 'Actual'
hdr_cells[3].text = 'Basis'
hdr_cells[4].text = 'Status'

metrics_data = [
    ("Source Support", ">= 85%", f"{summary['overall_metrics']['source_support_pct']}%", "judge", "Pass" if summary['overall_metrics']['source_support_pct'] >= 85 else "Fail"),
    ("Relation Precision", ">= 80%", f"{final_precision}%", verdict_basis, "Pass" if final_precision >= 80 else "Fail"),
    ("Entity Grounding", ">= 90%", f"{overall_entity_grounding}%", "graph", "Pass" if overall_entity_grounding >= 90 else "Fail"),
    ("Duplicate Rate", "<= 5%", f"{overall_duplicate_rate}%", "graph", "Pass" if overall_duplicate_rate <= 5 else "Fail"),
    ("Orphan Rate", "<= 10%", f"{overall_orphan_rate}%", "graph", "Pass" if overall_orphan_rate <= 10 else "Fail"),
]

for metric, target, actual, basis, status in metrics_data:
    row_cells = table.add_row().cells
    row_cells[0].text = metric
    row_cells[1].text = target
    row_cells[2].text = actual
    row_cells[3].text = basis
    row_cells[4].text = status

doc.add_paragraph()
doc.add_paragraph(f"Final verdict based on: {verdict_basis} precision")

# Save
output_path = EVAL_DIR / "EVALUATION_REPORT.docx"
doc.save(output_path)
print(f"DOCX report saved: {output_path}")
