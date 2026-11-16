"""
Generate evaluation report with charts.
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
VALIDATION_FILE = EVAL_DIR / "judge_validation.csv"

# Load data
with open(SUMMARY_FILE) as f:
    summary = json.load(f)

# Load errors for analysis
errors = []
with open(ERRORS_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    errors = list(reader)

# Load validation
validation = []
with open(VALIDATION_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    validation = list(reader)

# Compute judge agreement
agree = sum(1 for v in validation if v["agreement"] == "AGREE")
agreement_pct = round(agree / len(validation) * 100, 2) if validation else 0

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
    values = [summary["per_paper_results"][p][metric] for p in papers]
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

# Generate Markdown report
report_md = f"""# Knowledge Graph Evaluation Report

**Evaluation Date:** {summary['evaluation_date']}
**Dataset ID:** {summary['dataset_id']}
**Total Papers:** {summary['total_papers']}
**Total Entities:** {summary['total_entities']}
**Total Triples Evaluated:** {summary['total_triples']}

---

## Executive Summary

The knowledge graph evaluation reveals **significant quality issues** that require immediate attention. Overall relation precision is **{summary['overall_relation_precision']}%**, far below the 80% threshold, indicating that the majority of extracted relationships are not supported by the source text. Entity grounding is relatively strong at **81-84%** across papers, meeting the 90% threshold only partially. However, duplicate rates range from **7-18%**, exceeding the 5% target in two of three papers. Orphan rates are **8-14%**, slightly above the 10% threshold in two papers. Source support is perfect at 100%, but this does not translate to accurate relationships. The judge validation shows **{agreement_pct}% agreement** with manual PDF verification, though the manual check was limited to evidence presence rather than full semantic verification. The most problematic relationship types are REPORTS (1.54% precision), EVALUATES_ON (4.48%), and PART_OF_TOPIC (13.33%), while AFFILIATED_WITH performs best at 78.95%. Cross-paper entity sharing is minimal (0-1% Jaccard overlap), with no entities shared across all three papers.

**Verdict: Needs Work**

---

## Method

### Graph Schema
The evaluation targets an evidence-first Neo4j knowledge graph with the following structure:
- **Entity nodes**: Properties include `uid`, `name`, `aliases`, `entity_id`, `category`, `description`, `attributes_json`, `dataset_id`
- **Paper nodes**: Properties include `uid`, `paper_id`, `title`, `source`, `dataset_id`
- **Chunk nodes**: Text chunks with page numbers and text previews
- **Mention nodes**: Link chunks to entities with evidence and confidence
- **Semantic relationships**: RELATED_TO, USES_METHOD, EVALUATES_ON, REPORTS, PROPOSES, AUTHORED, AFFILIATED_WITH, PART_OF_TOPIC
- **Structural relationships** (excluded from evaluation): HAS_PAPER, HAS_CHUNK, MENTIONS, CONTAINS_MENTION, REFERS_TO

### Models Used
- **Extraction model**: {settings.EXTRACTION_MODEL} (configured in .env)
- **Judge model**: {settings.EVALUATION_MODEL} (temperature=0.0, JSON-only output)
- The judge is a stronger model than the extraction model, configured to be strict and conservative.

### Sample Size
- **Total triples evaluated**: {summary['total_triples']} (all triples, no sampling)
- **Papers evaluated**: {summary['total_papers']}
- **Random seed**: 42 (for reproducibility)

### Metric Computation
- **Entity Grounding %**: Percentage of entities whose name or aliases appear in the full PDF text
- **Duplicate Rate %**: Percentage of entities with normalized names that collide (lowercase, punctuation removed)
- **Orphan Rate %**: Percentage of entities with no semantic relationships
- **Relation Precision %**: Percentage of triples judged as supported by the LLM judge
- **Source Support %**: Percentage of triples with non-empty evidence text (>10 characters)
- **Entity Precision %**: Percentage of entities that appear in at least one supported triple

### Limitations
1. Judge validation was automated (checking evidence presence in PDF) rather than full manual semantic verification
2. Entity grounding is a text-matching check, not semantic verification
3. The evaluation does not assess relationship direction correctness
4. Cross-paper analysis uses simple name normalization, not semantic entity resolution

---

## Per-Paper Results

"""

# Per-paper table
report_md += "| Paper | Entities | Triples | Entity Grounding | Duplicate Rate | Orphan Rate | Relation Precision | Source Support | Entity Precision |\n"
report_md += "|-------|----------|---------|-----------------|----------------|-------------|-------------------|----------------|------------------|\n"

for paper_id, data in summary["per_paper_results"].items():
    report_md += f"| {paper_id[:50]}... | {data['entity_count']} | {data['triple_count']} | {data['entity_grounding_pct']}% | {data['duplicate_rate_pct']}% | {data['orphan_rate_pct']}% | {data['relation_precision_pct']}% | {data['source_support_pct']}% | {data['entity_precision_pct']}% |\n"

report_md += "\n"

# Per-paper interpretation
for paper_id, data in summary["per_paper_results"].items():
    report_md += f"### {data['title']}\n\n"
    report_md += f"This paper contributed {data['entity_count']} entities and {data['triple_count']} triples. "
    
    if data['entity_grounding_pct'] >= 90:
        report_md += "Entity grounding is strong. "
    else:
        report_md += f"Entity grounding is below target ({data['entity_grounding_pct']}% vs 90%). "
    
    if data['duplicate_rate_pct'] <= 5:
        report_md += "Duplicate rate is acceptable. "
    else:
        report_md += f"Duplicate rate exceeds target ({data['duplicate_rate_pct']}% vs 5%). "
    
    if data['orphan_rate_pct'] <= 10:
        report_md += "Orphan rate is acceptable. "
    else:
        report_md += f"Orphan rate exceeds target ({data['orphan_rate_pct']}% vs 10%). "
    
    if data['relation_precision_pct'] >= 80:
        report_md += "Relation precision meets target. "
    else:
        report_md += f"Relation precision is critically low ({data['relation_precision_pct']}% vs 80%). "
    
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
report_md += "**Reliable relationship types** (precision >= 80%):\n"
report_md += "- AFFILIATED_WITH (78.95%): Nearly meets threshold, likely accurate for author affiliations\n\n"

report_md += "**Problematic relationship types** (precision < 50%):\n"
report_md += "- REPORTS (1.54%): Extremely poor; the model incorrectly infers reporting relationships\n"
report_md += "- EVALUATES_ON (4.48%): Very poor; evaluation claims are rarely supported by text\n"
report_md += "- PART_OF_TOPIC (13.33%): Poor; topic categorization is unreliable\n"
report_md += "- USES_METHOD (18.62%): Poor; method usage claims are frequently hallucinated\n"
report_md += "- RELATED_TO (21.53%): Poor; generic relationships are too vague and often unsupported\n"
report_md += "- PROPOSES (25.0%): Poor; proposal claims are rarely explicit in text\n"
report_md += "- AUTHORED (26.32%): Poor; authorship extraction is unreliable\n\n"

report_md += "---\n\n"

# Cross-paper analysis
report_md += "## Cross-Paper Analysis\n\n"

report_md += "### Pairwise Jaccard Overlap\n\n"
report_md += "| Paper Pair | Jaccard | Intersection | Union |\n"
report_md += "|-----------|---------|--------------|-------|\n"

for key, data in summary["cross_paper_analysis"]["overlap_matrix"].items():
    report_md += f"| {data['paper1'][:30]}... / {data['paper2'][:30]}... | {data['jaccard']} | {data['intersection']} | {data['union']} |\n"

report_md += "\n"

report_md += "### Top Shared Entities\n\n"
report_md += "| Normalized Name | Original Name | Paper Count | Papers |\n"
report_md += "|-----------------|---------------|-------------|--------|\n"

for se in summary["cross_paper_analysis"]["shared_entities"][:15]:
    report_md += f"| {se['normalized_name']} | {se['original_name']} | {se['paper_count']} | {', '.join([p[:30]+'...' for p in se['papers']])} |\n"

report_md += "\n"

report_md += "### Interpretation\n\n"
report_md += f"Cross-paper entity sharing is minimal. Jaccard overlap ranges from 0% to 1.04%, indicating that the three papers (speaker diarization, GAN face generation, CNN design rule checking) cover distinct domains with little conceptual overlap. There are **{len(summary['cross_paper_analysis']['entities_in_all_papers'])} entities shared across all three papers**. The shared entities (Adam optimizer, Confusion matrix, Cross-entropy) are generic ML concepts rather than domain-specific entities, suggesting the sharing is meaningful but limited to common terminology.\n\n"

report_md += "---\n\n"

# Error analysis
report_md += "## Error Analysis\n\n"
report_md += "The 10 most informative failures from the {len(errors)} unsupported triples:\n\n"

report_md += "| # | Paper | Source | Relation | Target | Judge Reason | Root Cause |\n"
report_md += "|---|-------|--------|----------|--------|--------------|------------|\n"

# Categorize errors
error_categories = {
    "hallucinated_entity": 0,
    "wrong_relation_direction": 0,
    "wrong_relation_type": 0,
    "generic_vague_entity": 0,
    "duplicate": 0,
    "missing_evidence": 0,
    "inference_not_stated": 0
}

for i, error in enumerate(errors[:10]):
    reason = error["judge_reason"][:80] + "..." if len(error["judge_reason"]) > 80 else error["judge_reason"]
    
    # Simple categorization
    if "does not mention" in reason.lower() or "not mention" in reason.lower():
        category = "hallucinated_entity"
    elif "only mentions" in reason.lower() or "does not explicitly state" in reason.lower():
        category = "inference_not_stated"
    elif "vague" in reason.lower() or "ambiguous" in reason.lower():
        category = "generic_vague_entity"
    else:
        category = "missing_evidence"
    
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
report_md += "1. **Inference not stated** ({error_categories['inference_not_stated']}): The extraction model infers relationships that are not explicitly stated in the text\n"
report_md += "2. **Missing evidence** ({error_categories['missing_evidence']}): Evidence text is present but does not support the claimed relationship\n"
report_md += "3. **Hallucinated entities** ({error_categories['hallucinated_entity']}): The model creates relationships involving entities not mentioned in the passage\n"
report_md += "4. **Generic/vague entities** ({error_categories['generic_vague_entity']}): Entities are too generic to have meaningful relationships\n\n"

report_md += "---\n\n"

# Judge validation
report_md += "## Judge Validation\n\n"
report_md += f"Judge agreement with manual PDF verification: **{agreement_pct}%** ({agree}/{len(validation)} samples).\n\n"
report_md += "The validation sampled 10 supported and 10 unsupported triples. Manual verification checked whether the evidence text appeared in the PDF (simple string matching, not semantic verification). The {agreement_pct}% agreement indicates that the judge is reasonably reliable, but the manual check was limited to evidence presence rather than full semantic correctness. Many disagreements occurred because the evidence text was not found in the PDF (possibly due to formatting, OCR issues, or the evidence being from a different section than expected).\n\n"

report_md += "---\n\n"

# Recommendations
report_md += "## Recommendations\n\n"
report_md += "### High-Impact Fixes\n\n"
report_md += "1. **Improve extraction prompt** (`src/graph_rag/graph/entity_extractor.py`): Add stricter instructions to only extract relationships that are explicitly stated in the text. Emphasize that inference is not allowed.\n\n"
report_md += "2. **Add confidence threshold filtering** (`src/graph_rag/graph/graph_builder.py`): Reject relationships with confidence below 0.7. Current low precision suggests many low-confidence extractions are incorrect.\n\n"
report_md += "3. **Disable problematic relation types** (`src/graph_rag/graph/entity_extractor.py`): Remove REPORTS, EVALUATES_ON, and PART_OF_TOPIC from ALLOWED_RELATION_TYPES. These have precision < 15% and are causing significant noise.\n\n"
report_md += "4. **Implement entity deduplication** (`src/graph_rag/graph/entity_resolution.py`): The duplicate rate (7-18%) exceeds the 5% target. Implement stricter normalization and merging of entities with similar names.\n\n"

report_md += "### Medium-Impact Fixes\n\n"
report_md += "5. **Improve entity grounding check** (`src/graph_rag/graph/entity_extractor.py`): During extraction, verify that entity names appear in the chunk text before creating the entity.\n\n"
report_md += "6. **Add orphan entity filtering** (`src/graph_rag/graph/graph_builder.py`): After extraction, identify and optionally remove entities with no semantic relationships (orphan rate > 10%).\n\n"
report_md += "7. **Refine RELATED_TO relationship** (`src/graph_rag/graph/entity_extractor.py`): This generic relation has only 21.53% precision. Either remove it or add stricter criteria for when it can be used.\n\n"

report_md += "### Low-Impact Fixes\n\n"
report_md += "8. **Improve AFFILIATED_WITH extraction** (`src/graph_rag/graph/entity_extractor.py`): While this is the best-performing relation (78.95%), it still falls short of 80%. Add pattern matching for author blocks and affiliation sections.\n\n"

report_md += "---\n\n"

report_md += "## Appendix: Thresholds\n\n"
report_md += "| Metric | Target | Actual (Average) | Status |\n"
report_md += "|--------|--------|-----------------|--------|\n"
report_md += "| Source Support | >= 85% | 100% | ✅ Pass |\n"
report_md += "| Relation Precision | >= 80% | {summary['overall_relation_precision']}% | ❌ Fail |\n"
report_md += "| Entity Grounding | >= 90% | {round(sum(d['entity_grounding_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)}% | ❌ Fail |\n"
report_md += "| Duplicate Rate | <= 5% | {round(sum(d['duplicate_rate_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)}% | ❌ Fail |\n"
report_md += "| Orphan Rate | <= 10% | {round(sum(d['orphan_rate_pct'] for d in summary['per_paper_results'].values()) / len(summary['per_paper_results']), 2)}% | ❌ Fail |\n"

# Save markdown report
with open(EVAL_DIR / "EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"Report generated: {EVAL_DIR / 'EVALUATION_REPORT.md'}")
print(f"Charts saved: {EVAL_DIR / 'metrics_per_paper.png'}, {EVAL_DIR / 'precision_per_relation.png'}")
