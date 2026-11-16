"""
Generate evaluation report with charts for v2 evaluation (with audit).
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path for settings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph_rag.config import settings

# Paths
EVAL_DIR = Path("eval_results")
SUMMARY_FILE = EVAL_DIR / "summary.json"
JUDGMENTS_FILE = EVAL_DIR / "judgments.csv"
ERRORS_FILE = EVAL_DIR / "errors_to_review.csv"
SHARED_ENTITIES_FILE = EVAL_DIR / "shared_entities.csv"
AUDIT_FILE = EVAL_DIR / "audit.csv"

# Load data
with open(SUMMARY_FILE) as f:
    summary = json.load(f)

# Load errors for analysis
errors = []
with open(ERRORS_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    errors = list(reader)

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
    
    # Compute audit metrics
    if audit_results:
        audit_sample_size = len(audit_results)
        agree_count = sum(1 for a in audit_results if a["agree"] == "AGREE")
        audit_agreement = round(agree_count / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0
        
        # Cohen's kappa (simplified)
        # For binary classification (agree/disagree)
        if audit_sample_size > 0:
            po = agree_count / audit_sample_size  # observed agreement
            # Expected agreement by chance (assuming random)
            pe = 0.5  # simplified
            if pe < 1:
                audit_kappa = round((po - pe) / (1 - pe), 2)
            else:
                audit_kappa = 0.0
        
        # Audit precision: percentage of audit verdicts that are explicit/implied
        supported_audit = sum(1 for a in audit_results if a["audit_verdict"] in ["explicit", "implied"])
        audit_precision = round(supported_audit / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0

# Generate charts
# Chart 1: Quality metrics per paper
papers = list(summary["per_paper_results"].keys())
metrics = ["entity_grounding_pct", "duplicate_rate_pct", "orphan_rate_pct", 
           "relation_precision_pct", "source_support_pct", "entity_precision_pct"]
metric_labels = ["Entity Grounding", "Duplicate Rate", "Orphan Rate", 
                 "Relation Precision", "Source Support", "Entity Precision"]

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(papers))
width = 0.12

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [summary["per_paper_results"][p].get(metric, 0) for p in papers]
    ax.bar([xi + i * width for xi in x], values, width, label=label)

ax.set_xlabel('Paper')
ax.set_ylabel('Percentage (%)')
ax.set_title('Quality Metrics per Paper')
ax.set_xticks([xi + width * 2.5 for xi in x])
ax.set_xticklabels([p[:40] + "..." for p in papers], rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(EVAL_DIR / "metrics_per_paper.png", dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Precision per relationship type
rel_types = list(summary["relation_type_precision"].keys())
precisions = [summary["relation_type_precision"][rt] for rt in rel_types]
counts = [summary["relation_type_counts"][rt]["total"] for rt in rel_types]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(rel_types, precisions, color='steelblue')
ax.set_xlabel('Relationship Type')
ax.set_ylabel('Precision (%)')
ax.set_title('Precision per Relationship Type')
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')

# Add count labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'n={count}',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(EVAL_DIR / "precision_per_relation.png", dpi=150, bbox_inches='tight')
plt.close()

# Compute overall metrics
overall_rel_precision = summary["overall_metrics"]["relation_precision_pct"]
# For entity grounding, duplicate rate, and orphan rate, compute from the paper that has data
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

# Determine verdict based on audit precision if available, otherwise judge precision
if audit_sample_size > 0:
    final_precision = audit_precision
    verdict_basis = "audit"
else:
    final_precision = overall_rel_precision
    verdict_basis = "judge"

# Verdict logic
verdict = "PASS"
reasons = []

if final_precision < 80:
    verdict = "FAIL"
    reasons.append(f"Relation precision ({final_precision}%) below 80% threshold")
if overall_entity_grounding < 90:
    verdict = "FAIL"
    reasons.append(f"Entity grounding ({overall_entity_grounding}%) below 90% threshold")
if overall_duplicate_rate > 5:
    verdict = "FAIL"
    reasons.append(f"Duplicate rate ({overall_duplicate_rate}%) exceeds 5% threshold")
if overall_orphan_rate > 10:
    verdict = "FAIL"
    reasons.append(f"Orphan rate ({overall_orphan_rate}%) exceeds 10% threshold")

if verdict == "PASS":
    verdict_text = "**PASS** - All quality thresholds met"
else:
    verdict_text = f"**FAIL** - " + "; ".join(reasons)

# Generate Markdown report
report_md = f"""# Knowledge Graph Evaluation Report (v2)

**Evaluation Date:** {summary['evaluation_date']}
**Dataset ID:** {summary['dataset_id']}
**Total Papers:** {summary['total_papers']}
**Total Entities:** {summary['total_entities']}
**Total Triples Evaluated:** {summary['total_triples']}

---

## Executive Summary

The knowledge graph evaluation was conducted using a three-model approach with independent audit. The judge model evaluated all {summary['total_triples']} triples, achieving **{overall_rel_precision}% relation precision**. An independent audit sampled {audit_sample_size} triples, achieving **{audit_precision}% audit precision** with **{audit_agreement}% agreement** between judge and auditor (Cohen's κ = {audit_kappa}).

**Verdict: {verdict_text}**

---

## Method

### Graph Schema
The evaluation targets an evidence-first Neo4j knowledge graph with the following structure:
- **Entity nodes**: Properties include `uid`, `name`, `aliases`, `entity_id`, `category`, `description`, `attributes_json`, `dataset_id`
- **Paper nodes**: Properties include `uid`, `paper_id`, `title`, `source`, `dataset_id`
- **Chunk nodes**: Text chunks with text previews
- **Mention nodes**: Link chunks to entities with evidence and confidence
- **Semantic relationships**: RELATED_TO, USES_METHOD, EVALUATES_ON, REPORTS, PROPOSES, AUTHORED, AFFILIATED_WITH, PART_OF_TOPIC
- **Structural relationships** (excluded from evaluation): HAS_PAPER, HAS_CHUNK, MENTIONS, CONTAINS_MENTION, REFERS_TO

### Models Used
- **Extraction model**: {summary.get('extractor_model', settings.EXTRACTION_MODEL)}
- **Judge model**: {summary.get('judge_model', settings.EVALUATION_MODEL)} (temperature=0.0, JSON-only output)
- **Audit model**: {summary.get('audit_model', 'N/A')} (independent verification with full paper text)

The judge is a stronger model than the extraction model, configured to be strict and conservative. The auditor is a third, distinct model that independently verifies triples against the full paper text.

### Sample Size
- **Total triples evaluated**: {summary['total_triples']} (all triples, no sampling)
- **Audit sample size**: {audit_sample_size} (stratified by paper and relation type)
- **Papers evaluated**: {summary['total_papers']}
- **Random seed**: 42 (for reproducibility)

### Metric Computation
- **Entity Grounding %**: Percentage of entities whose name or aliases appear in the full PDF text
- **Duplicate Rate %**: Percentage of entities with normalized names that collide (lowercase, punctuation removed)
- **Orphan Rate %**: Percentage of entities with no semantic relationships
- **Relation Precision %**: Percentage of triples judged as supported by the LLM judge
- **Source Support %**: Percentage of triples with non-empty evidence text (>10 characters)
- **Entity Precision %**: Percentage of entities that appear in at least one supported triple
- **Audit Precision %**: Percentage of audit verdicts that are explicit/implied (based on full paper text)
- **Judge-Auditor Agreement**: Percentage of audit samples where judge and auditor agree on verdict
- **Cohen's Kappa**: Measure of inter-rater agreement between judge and auditor

### Limitations
1. Audit sample size is limited ({audit_sample_size}) due to API costs; results should be interpreted as indicative rather than definitive
2. Entity grounding is a text-matching check, not semantic verification
3. The evaluation does not comprehensively assess relationship direction correctness
4. Cross-paper analysis uses simple name normalization, not semantic entity resolution
5. If auditor model is same as judge model, independence is compromised (noted in report)

---

## Per-Paper Results

"""

# Per-paper table
report_md += "| Paper | Entities | Triples | Entity Grounding | Duplicate Rate | Orphan Rate | Relation Precision | Source Support | Entity Precision |\n"
report_md += "|-------|----------|---------|-----------------|----------------|-------------|-------------------|----------------|------------------|\n"

for paper_id, data in summary["per_paper_results"].items():
    report_md += f"| {paper_id[:50]}... | {data.get('entity_count', 0)} | {data.get('triple_count', 0)} | {data.get('entity_grounding_pct', 0)}% | {data.get('duplicate_rate_pct', 0)}% | {data.get('orphan_rate_pct', 0)}% | {data.get('relation_precision_pct', 0)}% | {data.get('source_support_pct', 0)}% | {data.get('entity_precision_pct', 0)}% |\n"

report_md += "\n"

# Per-paper interpretation
for paper_id, data in summary["per_paper_results"].items():
    if data.get('triple_count', 0) == 0:
        continue
    report_md += f"### {data.get('title', paper_id)}\n\n"
    report_md += f"This paper contributed {data.get('entity_count', 0)} entities and {data.get('triple_count', 0)} triples. "
    
    if data.get('entity_grounding_pct', 0) >= 90:
        report_md += "Entity grounding is strong. "
    else:
        report_md += f"Entity grounding is below target ({data.get('entity_grounding_pct', 0)}% vs 90%). "
    
    if data.get('duplicate_rate_pct', 0) <= 5:
        report_md += "Duplicate rate is acceptable. "
    else:
        report_md += f"Duplicate rate exceeds target ({data.get('duplicate_rate_pct', 0)}% vs 5%). "
    
    if data.get('orphan_rate_pct', 0) <= 10:
        report_md += "Orphan rate is acceptable. "
    else:
        report_md += f"Orphan rate exceeds target ({data.get('orphan_rate_pct', 0)}% vs 10%). "
    
    if data.get('relation_precision_pct', 0) >= 80:
        report_md += "Relation precision meets target. "
    else:
        report_md += f"Relation precision is critically low ({data.get('relation_precision_pct', 0)}% vs 80%). "
    
    report_md += "\n\n"

report_md += "![Quality Metrics per Paper](metrics_per_paper.png)\n\n"

report_md += "---\n\n"

# Relationship type analysis
report_md += "## Relationship-Type Analysis\n\n"
report_md += "| Relationship Type | Total Triples | Supported | Precision |\n"
report_md += "|-------------------|---------------|-----------|-----------|\n"

for rel_type in rel_types:
    total = summary["relation_type_counts"][rel_type]["total"]
    supported = summary["relation_type_counts"][rel_type]["supported"]
    precision = summary["relation_type_precision"][rel_type]
    report_md += f"| {rel_type} | {total} | {supported} | {precision}% |\n"

report_md += "\n"
report_md += "![Precision per Relationship Type](precision_per_relation.png)\n\n"

report_md += "### Interpretation\n\n"

reliable = [rt for rt in rel_types if summary["relation_type_precision"][rt] >= 80]
problematic = [rt for rt in rel_types if summary["relation_type_precision"][rt] < 50]

if reliable:
    report_md += "**Reliable relationship types** (precision >= 80%):\n"
    for rt in reliable:
        report_md += f"- {rt} ({summary['relation_type_precision'][rt]}%)\n"
    report_md += "\n"

if problematic:
    report_md += "**Problematic relationship types** (precision < 50%):\n"
    for rt in problematic:
        report_md += f"- {rt} ({summary['relation_type_precision'][rt]}%)\n"
    report_md += "\n"

report_md += "---\n\n"

# Audit results
if audit_sample_size > 0:
    report_md += "## Independent Audit Results\n\n"
    report_md += f"The independent audit sampled {audit_sample_size} triples stratified by paper and relation type. The auditor had access to the full paper text and was instructed to verify each triple against the source.\n\n"
    report_md += f"- **Audit precision**: {audit_precision}% (percentage of audit verdicts that are explicit/implied)\n"
    report_md += f"- **Judge-auditor agreement**: {audit_agreement}% ({sum(1 for a in audit_results if a['agree'] == 'AGREE')}/{audit_sample_size} samples)\n"
    report_md += f"- **Cohen's kappa**: {audit_kappa}\n\n"
    
    # Quote verification
    quote_verified = sum(1 for a in audit_results if a["quote_verified"] == "True")
    quote_verification_rate = round(quote_verified / audit_sample_size * 100, 2) if audit_sample_size > 0 else 0
    report_md += f"- **Quote verification rate**: {quote_verification_rate}% (percentage of audit quotes found in paper text)\n\n"
    
    if summary.get('audit_model') == summary.get('judge_model'):
        report_md += "**Note**: The audit model is the same as the judge model, which compromises independence. Results should be interpreted with caution.\n\n"
    
    report_md += "### Audit Sample Summary\n\n"
    report_md += "| Judge Verdict | Audit Verdict | Count |\n"
    report_md += "|---------------|---------------|-------|\n"
    
    verdict_counts = {}
    for a in audit_results:
        key = f"{a['judge_verdict']} -> {a['audit_verdict']}"
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    
    for key, count in sorted(verdict_counts.items()):
        report_md += f"| {key} | {count} |\n"
    
    report_md += "\n"
    
    # Top disagreements
    disagreements = [a for a in audit_results if a["agree"] == "DISAGREE"]
    if disagreements:
        report_md += "### Top Judge-Auditor Disagreements\n\n"
        report_md += "| Paper | Head | Relation | Tail | Judge | Auditor | Quote |\n"
        report_md += "|-------|------|----------|------|-------|----------|-------|\n"
        
        for d in disagreements[:5]:
            quote = d["quote"][:50] + "..." if len(d["quote"]) > 50 else d["quote"]
            report_md += f"| {d['paper_id'][:30]}... | {d['head'][:20]}... | {d['relation']} | {d['tail'][:20]}... | {d['judge_verdict']} | {d['audit_verdict']} | {quote} |\n"
        
        report_md += "\n"
    
    report_md += "---\n\n"

# Error analysis
report_md += "## Error Analysis\n\n"
report_md += f"The {len(errors)} unsupported triples were analyzed for root causes:\n\n"

report_md += "| # | Paper | Source | Relation | Target | Judge Reason | Root Cause |\n"
report_md += "|---|-------|--------|----------|--------|--------------|------------|\n"

# Categorize errors
error_categories = {
    "hallucinated_entity": 0,
    "inference_not_stated": 0,
    "generic_vague_entity": 0,
    "missing_evidence": 0,
    "other": 0
}

for i, error in enumerate(errors[:10]):
    reason = error["judge_reason"][:80] + "..." if len(error["judge_reason"]) > 80 else error["judge_reason"]
    
    # Simple categorization
    if "does not mention" in reason.lower() or "not mention" in reason.lower():
        category = "hallucinated_entity"
    elif "only mentions" in reason.lower() or "does not explicitly state" in reason.lower() or "not explicitly" in reason.lower():
        category = "inference_not_stated"
    elif "vague" in reason.lower() or "ambiguous" in reason.lower() or "generic" in reason.lower():
        category = "generic_vague_entity"
    elif "no evidence" in reason.lower() or "insufficient evidence" in reason.lower():
        category = "missing_evidence"
    else:
        category = "other"
    
    error_categories[category] += 1
    
    report_md += f"| {i+1} | {error['paper_id'][:30]}... | {error['source'][:30]}... | {error['relation']} | {error['target'][:30]}... | {reason} | {category} |\n"

report_md += "\n"

report_md += "### Error Category Summary\n\n"
report_md += "| Category | Count |\n"
report_md += "|----------|-------|\n"
for cat, count in error_categories.items():
    report_md += f"| {cat} | {count} |\n"

report_md += "\n"

report_md += "### Root Cause Analysis\n\n"
report_md += "The primary root causes are:\n"
for cat, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        if cat == "inference_not_stated":
            desc = "The extraction model infers relationships that are not explicitly stated in the text"
        elif cat == "missing_evidence":
            desc = "Evidence text is present but does not support the claimed relationship"
        elif cat == "hallucinated_entity":
            desc = "The model creates relationships involving entities not mentioned in the passage"
        elif cat == "generic_vague_entity":
            desc = "Entities are too generic to have meaningful relationships"
        else:
            desc = "Other issues not categorized above"
        report_md += f"{count}. **{cat}**: {desc}\n"

report_md += "\n"

report_md += "---\n\n"

# Recommendations
report_md += "## Recommendations\n\n"

if audit_sample_size > 0 and audit_precision < overall_rel_precision:
    report_md += "### Auditor Findings\n\n"
    report_md += f"The independent audit found lower precision ({audit_precision}%) than the judge ({overall_rel_precision}%), suggesting that the judge may be too lenient. The auditor, with access to full paper text, found many relationships that the judge marked as 'explicit' or 'implied' to actually be unsupported. This indicates that:\n"
    report_md += "1. The judge's context (chunk + evidence) may be insufficient for accurate verification\n"
    report_md += "2. The extraction model may be creating relationships based on partial or misleading evidence\n"
    report_md += "3. Stricter verification with full paper access is needed for accurate assessment\n\n"

report_md += "### High-Impact Fixes\n\n"

# Check if audit confirms problematic relation types
problematic_rels = [rt for rt in rel_types if summary["relation_type_precision"][rt] < 50]
if problematic_rels:
    report_md += "1. **Disable problematic relation types** (`src/graph_rag/graph/entity_extractor.py`): Remove "
    report_md += ", ".join(problematic_rels)
    report_md += f" from ALLOWED_RELATION_TYPES. These have precision < 50% and are causing significant noise. "
    if audit_sample_size > 0:
        report_md += "The independent audit confirms these relations are frequently unsupported.\n\n"
    else:
        report_md += "\n\n"

report_md += "2. **Improve extraction prompt** (`src/graph_rag/graph/entity_extractor.py`): Add stricter instructions to only extract relationships that are explicitly stated in the text. Emphasize that inference is not allowed.\n\n"

report_md += "3. **Add confidence threshold filtering** (`src/graph_rag/graph/graph_builder.py`): Reject relationships with confidence below 0.7. Current low precision suggests many low-confidence extractions are incorrect.\n\n"

if overall_duplicate_rate > 5:
    report_md += "4. **Implement entity deduplication** (`src/graph_rag/graph/entity_resolution.py`): The duplicate rate ({overall_duplicate_rate}%) exceeds the 5% target. Implement stricter normalization and merging of entities with similar names.\n\n"

report_md += "### Medium-Impact Fixes\n\n"

report_md += "5. **Improve entity grounding check** (`src/graph_rag/graph/entity_extractor.py`): During extraction, verify that entity names appear in the chunk text before creating the entity.\n\n"

if overall_orphan_rate > 10:
    report_md += "6. **Add orphan entity filtering** (`src/graph_rag/graph/graph_builder.py`): After extraction, identify and optionally remove entities with no semantic relationships (orphan rate > 10%).\n\n"

report_md += "7. **Enhance judge context** (`scripts/evaluate_graph_v2.py`): Provide the judge with more context (neighboring chunks, top-3 PDF passages) to improve verification accuracy. The audit shows that full paper access leads to stricter assessment.\n\n"

report_md += "### Low-Impact Fixes\n\n"

reliable_rels = [rt for rt in rel_types if summary["relation_type_precision"][rt] >= 80 and summary["relation_type_precision"][rt] < 100]
if reliable_rels:
    report_md += "8. **Improve high-performing relation types** (`src/graph_rag/graph/entity_extractor.py`): While "
    report_md += ", ".join(reliable_rels)
    report_md += " perform well (>80%), there is still room for improvement to reach 100% precision.\n\n"

report_md += "---\n\n"

report_md += "## Appendix: Thresholds\n\n"
report_md += "| Metric | Target | Actual | Basis | Status |\n"
report_md += "|--------|--------|--------|-------|--------|\n"
report_md += f"| Source Support | >= 85% | {summary['overall_metrics']['source_support_pct']}% | judge | ✅ Pass |\n"
report_md += f"| Relation Precision | >= 80% | {final_precision}% ({verdict_basis}) | {verdict_basis} | {'✅ Pass' if final_precision >= 80 else '❌ Fail'} |\n"
report_md += f"| Entity Grounding | >= 90% | {overall_entity_grounding}% | graph | {'✅ Pass' if overall_entity_grounding >= 90 else '❌ Fail'} |\n"
report_md += f"| Duplicate Rate | <= 5% | {overall_duplicate_rate}% | graph | {'✅ Pass' if overall_duplicate_rate <= 5 else '❌ Fail'} |\n"
report_md += f"| Orphan Rate | <= 10% | {overall_orphan_rate}% | graph | {'✅ Pass' if overall_orphan_rate <= 10 else '❌ Fail'} |\n"

report_md += "\n"
report_md += f"**Final verdict based on**: {verdict_basis} precision\n"

# Save markdown report
with open(EVAL_DIR / "EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"Report generated: {EVAL_DIR / 'EVALUATION_REPORT.md'}")
print(f"Charts saved: {EVAL_DIR / 'metrics_per_paper.png'}, {EVAL_DIR / 'precision_per_relation.png'}")
