"""
Knowledge Graph Evaluation Script

Evaluates the quality of the Neo4j knowledge graph built from research papers.
Metrics include entity grounding, duplicate rate, orphan rate, relation precision,
source support, and cross-paper entity sharing.

Usage:
    python scripts/evaluate_graph.py inspect          # Inspect schema
    python scripts/evaluate_graph.py evaluate --sample 0  # Evaluate all triples
    python scripts/evaluate_graph.py evaluate --sample 60  # Sample 60 per paper
"""

import argparse
import csv
import json
import logging
import random
import re
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_rag.config import settings
from graph_rag.llm.model_client import chat_model_client

# Project root (same as in config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# CONFIGURATION BLOCK - Update based on schema inspection
# ==============================================================================

CONFIG = {
    # Entity node properties
    "entity_label": "Entity",
    "entity_name_prop": "name",
    "entity_aliases_prop": "aliases",
    "entity_id_prop": "entity_id",
    
    # Paper node properties
    "paper_label": "Paper",
    "paper_uid_prop": "uid",
    "paper_title_prop": "title",
    "paper_id_prop": "paper_id",
    
    # Relationship evidence property
    "relation_evidence_prop": "evidence",
    "relation_confidence_prop": "confidence",
    "relation_chunk_id_prop": "chunk_id",
    "relation_paper_id_prop": "paper_id",
    
    # Structural relationships to EXCLUDE from evaluation
    "structural_rels": {
        "HAS_PAPER", "HAS_CHUNK", "MENTIONS", "CONTAINS_MENTION",
        "REFERS_TO"
    },
    
    # Semantic relationships to EVALUATE
    "semantic_rels": {
        "RELATED_TO", "USES_METHOD", "EVALUATES_ON", "REPORTS",
        "PROPOSES", "AUTHORED", "AFFILIATED_WITH", "PART_OF_TOPIC"
    },
    
    # Paper-to-Entity path for grounding check
    "paper_to_entity_path": "MENTIONS",
    
    # Judge model (should be stronger than extraction model)
    "judge_model": settings.EVALUATION_MODEL,
    "judge_temperature": 0.0,
    
    # Output directory
    "output_dir": PROJECT_ROOT / "eval_results",
    
    # Dataset ID filter
    "dataset_id": settings.DATASET_ID,
    
    # Random seed for reproducibility
    "seed": 42,
}

# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

random.seed(CONFIG["seed"])

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


# ==============================================================================
# NEO4J CONNECTION
# ==============================================================================

from neo4j import GraphDatabase

def get_driver():
    """Get Neo4j driver."""
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=30.0,
        )
        driver.verify_connectivity()
        return driver
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)


# ==============================================================================
# SCHEMA INSPECTION
# ==============================================================================

def inspect_schema():
    """Inspect the actual graph schema and print findings."""
    driver = get_driver()
    
    print("=" * 80)
    print("GRAPH SCHEMA INSPECTION")
    print("=" * 80)
    
    with driver.session() as session:
        # Get all node labels
        labels_result = session.run("""
            CALL db.labels() YIELD label
            RETURN label ORDER BY label
        """)
        labels = [record["label"] for record in labels_result]
        print(f"\nNode Labels: {labels}")
        
        # Get all relationship types
        rels_result = session.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN relationshipType ORDER BY relationshipType
        """)
        rel_types = [record["relationshipType"] for record in rels_result]
        print(f"\nRelationship Types: {rel_types}")
        
        # Sample Entity node properties
        entity_props = session.run("""
            MATCH (e:Entity {dataset_id: $dataset_id})
            RETURN e
            LIMIT 1
        """, dataset_id=CONFIG["dataset_id"])
        
        entity_record = entity_props.single()
        if entity_record:
            print(f"\nSample Entity Properties:")
            for key, value in entity_record["e"].items():
                if key not in ["dataset_id", "schema_version", "created_at"]:
                    print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
        
        # Sample Paper node properties
        paper_props = session.run("""
            MATCH (p:Paper {dataset_id: $dataset_id})
            RETURN p
            LIMIT 1
        """, dataset_id=CONFIG["dataset_id"])
        
        paper_record = paper_props.single()
        if paper_record:
            print(f"\nSample Paper Properties:")
            for key, value in paper_record["p"].items():
                if key not in ["dataset_id", "schema_version", "created_at"]:
                    print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
        
        # Sample relationship properties
        rel_props = session.run("""
            MATCH (a)-[r]->(b)
            WHERE a.dataset_id = $dataset_id
            RETURN r, type(r) as rel_type
            LIMIT 1
        """, dataset_id=CONFIG["dataset_id"])
        
        rel_record = rel_props.single()
        if rel_record:
            print(f"\nSample Relationship Properties (type: {rel_record['rel_type']}):")
            for key, value in rel_record["r"].items():
                if key not in ["dataset_id", "extraction_version", "evidence_id"]:
                    print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
        
        # Count nodes and edges
        stats = session.run("""
            MATCH (n {dataset_id: $dataset_id})
            WITH count(n) as nodes
            OPTIONAL MATCH ()-[r {dataset_id: $dataset_id}]->()
            RETURN nodes, count(r) as edges
        """, dataset_id=CONFIG["dataset_id"])
        
        stat_record = stats.single()
        print(f"\nTotal Nodes: {stat_record['nodes']}")
        print(f"Total Edges: {stat_record['edges']}")
        
        # Count by paper
        paper_counts = session.run("""
            MATCH (p:Paper {dataset_id: $dataset_id})
            OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity {dataset_id: $dataset_id})
            WITH p, count(DISTINCT e) as entity_count
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk {dataset_id: $dataset_id})
            WITH p, entity_count, count(DISTINCT c) as chunk_count
            RETURN p.paper_id as paper_id, p.title as title, entity_count, chunk_count
            ORDER BY paper_id
        """, dataset_id=CONFIG["dataset_id"])
        
        print(f"\nPapers in graph:")
        for record in paper_counts:
            print(f"  {record['paper_id']}: {record['entity_count']} entities, {record['chunk_count']} chunks")
    
    driver.close()
    print("\n" + "=" * 80)
    print("Update CONFIG block in this script to match the actual schema.")
    print("=" * 80)


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def get_all_papers(driver) -> List[Dict[str, Any]]:
    """Get all papers with their UIDs and metadata."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper {dataset_id: $dataset_id})
            RETURN p.uid as uid, p.paper_id as paper_id, p.title as title
            ORDER BY p.paper_id
        """, dataset_id=CONFIG["dataset_id"])
        return [dict(record) for record in result]


def get_entities_for_paper(driver, paper_uid: str) -> List[Dict[str, Any]]:
    """Get all entities mentioned by a paper."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paper {uid: $paper_uid, dataset_id: $dataset_id})-[:MENTIONS]->(e:Entity {dataset_id: $dataset_id})
            RETURN e.uid as uid, e.entity_id as entity_id, e.name as name, 
                   e.category as category, e.aliases as aliases
            ORDER BY e.name
        """, paper_uid=paper_uid, dataset_id=CONFIG["dataset_id"])
        return [dict(record) for record in result]


def get_semantic_triples_for_paper(driver, paper_id: str) -> List[Dict[str, Any]]:
    """Get all semantic relationships (triples) for a paper."""
    semantic_rels_list = list(CONFIG["semantic_rels"])
    
    with driver.session() as session:
        # Build dynamic query with multiple relationship types
        query = f"""
            MATCH (s:Entity {{dataset_id: $dataset_id}})-[r:{'|'.join(semantic_rels_list)}]->(t:Entity {{dataset_id: $dataset_id}})
            WHERE r.dataset_id = $dataset_id AND r.paper_id = $paper_id
            RETURN s.name as source, type(r) as relation, t.name as target,
                   s.category as source_category, t.category as target_category,
                   r.evidence as evidence, r.confidence as confidence,
                   r.chunk_id as chunk_id, r.paper_id as paper_id,
                   s.entity_id as source_entity_id, t.entity_id as target_entity_id
        """
        result = session.run(query, dataset_id=CONFIG["dataset_id"], paper_id=paper_id)
        return [dict(record) for record in result]


def get_pdf_text(paper_id: str) -> str:
    """Extract full text from PDF for grounding checks."""
    # paper_id already includes .pdf extension
    pdf_path = settings.PDF_DIR / paper_id
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return ""
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def normalize_entity_name(name: str) -> str:
    """Normalize entity name for duplicate detection."""
    # Convert to lowercase
    normalized = name.lower()
    # Remove punctuation
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def compute_entity_grounding(entities: List[Dict[str, Any]], pdf_text: str) -> Tuple[int, int]:
    """Compute how many entities are grounded in the PDF text."""
    if not pdf_text:
        return 0, len(entities)
    
    pdf_lower = pdf_text.lower()
    grounded = 0
    
    for entity in entities:
        name = entity["name"].lower()
        aliases = entity.get("aliases") or []
        
        # Check if name or any alias appears in PDF
        if name in pdf_lower:
            grounded += 1
        else:
            for alias in aliases:
                if alias.lower() in pdf_lower:
                    grounded += 1
                    break
    
    return grounded, len(entities)


def compute_duplicate_rate(entities: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Compute duplicate entity rate after normalization."""
    normalized_names = {}
    duplicates = 0
    
    for entity in entities:
        norm_name = normalize_entity_name(entity["name"])
        if norm_name in normalized_names:
            duplicates += 1
        else:
            normalized_names[norm_name] = entity["name"]
    
    return duplicates, len(entities)


def compute_orphan_rate(entities: List[Dict[str, Any]], triples: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Compute percentage of entities with no semantic relationships."""
    entity_names_in_triples = set()
    
    for triple in triples:
        entity_names_in_triples.add(triple["source"])
        entity_names_in_triples.add(triple["target"])
    
    orphans = 0
    for entity in entities:
        if entity["name"] not in entity_names_in_triples:
            orphans += 1
    
    return orphans, len(entities)


# ==============================================================================
# LLM JUDGE
# ==============================================================================

JUDGE_PROMPT_TEMPLATE = """You are a strict fact-checker evaluating knowledge graph triples extracted from research papers.

For each triple, determine if it is SUPPORTED by the provided text passage.
- A triple is SUPPORTED only if the passage explicitly states the relationship.
- Do NOT infer relationships that are not directly stated.
- If the passage is vague or the relationship is ambiguous, mark as UNSUPPORTED.

Triple: {source} --[{relation}]--> {target}
Passage: {evidence}

Return JSON only:
{{
  "is_supported": true/false,
  "reason": "brief explanation (1-2 sentences)"
}}
"""


def judge_triple(triple: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Use LLM judge to evaluate a single triple."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        source=triple["source"],
        relation=triple["relation"],
        target=triple["target"],
        evidence=triple.get("evidence", "")[:2000]
    )
    
    for attempt in range(max_retries):
        try:
            response = chat_model_client.generate(
                prompt=prompt,
                system_prompt="Return valid JSON only. Be strict and conservative.",
                temperature=CONFIG["judge_temperature"],
            )
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in judge response: {response[:200]}")
                return {"is_supported": False, "reason": "Invalid JSON response"}
            
            data = json.loads(json_match.group(0))
            return {
                "is_supported": bool(data.get("is_supported", False)),
                "reason": str(data.get("reason", ""))[:500]
            }
        except Exception as e:
            logger.warning(f"Judge attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {"is_supported": False, "reason": f"Judge error: {str(e)[:200]}"}
    
    return {"is_supported": False, "reason": "Max retries exceeded"}


def batch_judge_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Judge a batch of triples."""
    results = []
    total = len(triples)
    
    for i, triple in enumerate(triples):
        if i % 10 == 0:
            logger.info(f"Judging triple {i+1}/{total}")
        
        judgment = judge_triple(triple)
        results.append({
            **triple,
            "is_supported": judgment["is_supported"],
            "judge_reason": judgment["reason"]
        })
    
    return results


# ==============================================================================
# CROSS-PAPER ANALYSIS
# ==============================================================================

def analyze_cross_paper_entities(all_entities_by_paper: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze entity sharing across papers."""
    # Build normalized name to papers mapping
    name_to_papers: Dict[str, Set[str]] = {}
    name_to_original: Dict[str, str] = {}
    
    for paper_id, entities in all_entities_by_paper.items():
        for entity in entities:
            norm_name = normalize_entity_name(entity["name"])
            if norm_name not in name_to_papers:
                name_to_papers[norm_name] = set()
                name_to_original[norm_name] = entity["name"]
            name_to_papers[norm_name].add(paper_id)
    
    # Find shared entities
    shared_entities = []
    for norm_name, papers in name_to_papers.items():
        if len(papers) > 1:
            shared_entities.append({
                "normalized_name": norm_name,
                "original_name": name_to_original[norm_name],
                "paper_count": len(papers),
                "papers": sorted(list(papers))
            })
    
    # Sort by paper count (descending)
    shared_entities.sort(key=lambda x: x["paper_count"], reverse=True)
    
    # Compute pairwise Jaccard overlap
    paper_ids = sorted(list(all_entities_by_paper.keys()))
    overlap_matrix = {}
    
    for i, p1 in enumerate(paper_ids):
        for p2 in paper_ids[i+1:]:
            set1 = {normalize_entity_name(e["name"]) for e in all_entities_by_paper[p1]}
            set2 = {normalize_entity_name(e["name"]) for e in all_entities_by_paper[p2]}
            
            if set1 or set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
            else:
                jaccard = 0.0
            
            overlap_matrix[f"{p1}__{p2}"] = {
                "paper1": p1,
                "paper2": p2,
                "jaccard": round(jaccard, 4),
                "intersection": len(set1 & set2),
                "union": len(set1 | set2)
            }
    
    # Find entities in all papers
    all_papers_entities = [e for e in shared_entities if e["paper_count"] == len(paper_ids)]
    
    return {
        "shared_entities": shared_entities,
        "overlap_matrix": overlap_matrix,
        "entities_in_all_papers": all_papers_entities,
        "total_papers": len(paper_ids)
    }


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def run_evaluation(sample_size: int = 0):
    """Run the full graph evaluation."""
    logger.info("Starting graph evaluation...")
    logger.info(f"Sample size per paper: {sample_size if sample_size > 0 else 'ALL'}")
    
    driver = get_driver()
    
    # Get all papers
    papers = get_all_papers(driver)
    logger.info(f"Found {len(papers)} papers")
    
    if not papers:
        logger.error("No papers found in graph!")
        driver.close()
        return
    
    # Collect data for all papers
    all_data = {}
    all_entities_by_paper = {}
    all_triples_by_paper = {}
    
    for paper in papers:
        paper_uid = paper["uid"]
        paper_id = paper["paper_id"]
        
        logger.info(f"Processing paper: {paper_id}")
        
        # Get entities
        entities = get_entities_for_paper(driver, paper_uid)
        all_entities_by_paper[paper_id] = entities
        
        # Get semantic triples
        triples = get_semantic_triples_for_paper(driver, paper_id)
        
        # Sample if needed
        if sample_size > 0 and len(triples) > sample_size:
            triples = random.sample(triples, sample_size)
        
        all_triples_by_paper[paper_id] = triples
        
        # Get PDF text for grounding
        pdf_text = get_pdf_text(paper_id)
        
        # Compute metrics
        grounded, total_entities = compute_entity_grounding(entities, pdf_text)
        duplicates, _ = compute_duplicate_rate(entities)
        orphans, _ = compute_orphan_rate(entities, triples)
        
        all_data[paper_id] = {
            "paper_id": paper_id,
            "title": paper["title"],
            "entity_count": len(entities),
            "triple_count": len(triples),
            "grounded_entities": grounded,
            "entity_grounding_pct": round(grounded / total_entities * 100, 2) if total_entities > 0 else 0,
            "duplicate_entities": duplicates,
            "duplicate_rate_pct": round(duplicates / total_entities * 100, 2) if total_entities > 0 else 0,
            "orphan_entities": orphans,
            "orphan_rate_pct": round(orphans / total_entities * 100, 2) if total_entities > 0 else 0,
            "pdf_text_length": len(pdf_text)
        }
    
    # Judge all triples
    logger.info("Starting LLM judgment of triples...")
    all_judgments = []
    
    for paper_id, triples in all_triples_by_paper.items():
        logger.info(f"Judging triples for {paper_id} ({len(triples)} triples)")
        judgments = batch_judge_triples(triples)
        all_judgments.extend(judgments)
        
        # Update paper data with judgment metrics
        supported = sum(1 for j in judgments if j["is_supported"])
        all_data[paper_id]["supported_triples"] = supported
        all_data[paper_id]["relation_precision_pct"] = round(supported / len(judgments) * 100, 2) if judgments else 0
    
    # Compute source support (evidence present)
    for paper_id, triples in all_triples_by_paper.items():
        with_evidence = sum(1 for t in triples if t.get("evidence") and len(t["evidence"]) > 10)
        all_data[paper_id]["source_support_pct"] = round(with_evidence / len(triples) * 100, 2) if triples else 0
    
    # Compute entity precision (entities in supported triples)
    entity_precision_data = {}
    for judgment in all_judgments:
        if judgment["is_supported"]:
            entity_precision_data[judgment["source"]] = entity_precision_data.get(judgment["source"], 0) + 1
            entity_precision_data[judgment["target"]] = entity_precision_data.get(judgment["target"], 0) + 1
    
    for paper_id in all_data:
        entities_in_supported = sum(1 for e in all_entities_by_paper[paper_id] 
                                   if e["name"] in entity_precision_data)
        all_data[paper_id]["entity_precision_pct"] = round(
            entities_in_supported / len(all_entities_by_paper[paper_id]) * 100, 2
        ) if all_entities_by_paper[paper_id] else 0
    
    # Count triples by relation type
    relation_type_counts = {}
    relation_type_precision = {}
    
    for judgment in all_judgments:
        rel_type = judgment["relation"]
        if rel_type not in relation_type_counts:
            relation_type_counts[rel_type] = {"total": 0, "supported": 0}
        relation_type_counts[rel_type]["total"] += 1
        if judgment["is_supported"]:
            relation_type_counts[rel_type]["supported"] += 1
    
    for rel_type, counts in relation_type_counts.items():
        relation_type_precision[rel_type] = round(
            counts["supported"] / counts["total"] * 100, 2
        ) if counts["total"] > 0 else 0
    
    # Cross-paper analysis
    logger.info("Analyzing cross-paper entity sharing...")
    cross_paper = analyze_cross_paper_entities(all_entities_by_paper)
    
    driver.close()
    
    # Save results
    logger.info("Saving results...")
    
    # Summary JSON
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_id": CONFIG["dataset_id"],
        "sample_size_per_paper": sample_size,
        "total_papers": len(papers),
        "total_entities": sum(len(e) for e in all_entities_by_paper.values()),
        "total_triples": sum(len(t) for t in all_triples_by_paper.values()),
        "total_judgments": len(all_judgments),
        "overall_relation_precision": round(
            sum(1 for j in all_judgments if j["is_supported"]) / len(all_judgments) * 100, 2
        ) if all_judgments else 0,
        "per_paper_results": all_data,
        "relation_type_counts": relation_type_counts,
        "relation_type_precision": relation_type_precision,
        "cross_paper_analysis": cross_paper
    }
    
    with open(CONFIG["output_dir"] / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Judgments CSV
    with open(CONFIG["output_dir"] / "judgments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "source", "relation", "target", "is_supported",
            "judge_reason", "evidence", "confidence"
        ])
        writer.writeheader()
        for j in all_judgments:
            writer.writerow({
                "paper_id": j.get("paper_id", ""),
                "source": j["source"],
                "relation": j["relation"],
                "target": j["target"],
                "is_supported": j["is_supported"],
                "judge_reason": j["judge_reason"],
                "evidence": j.get("evidence", "")[:500],
                "confidence": j.get("confidence", 0)
            })
    
    # Errors to review (unsupported triples)
    errors = [j for j in all_judgments if not j["is_supported"]]
    with open(CONFIG["output_dir"] / "errors_to_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "source", "relation", "target", "judge_reason", "evidence"
        ])
        writer.writeheader()
        for e in errors:
            writer.writerow({
                "paper_id": e.get("paper_id", ""),
                "source": e["source"],
                "relation": e["relation"],
                "target": e["target"],
                "judge_reason": e["judge_reason"],
                "evidence": e.get("evidence", "")[:500]
            })
    
    # Shared entities CSV
    with open(CONFIG["output_dir"] / "shared_entities.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "normalized_name", "original_name", "paper_count", "papers"
        ])
        writer.writeheader()
        for se in cross_paper["shared_entities"]:
            writer.writerow({
                "normalized_name": se["normalized_name"],
                "original_name": se["original_name"],
                "paper_count": se["paper_count"],
                "papers": ", ".join(se["papers"])
            })
    
    # Cross-paper JSON
    with open(CONFIG["output_dir"] / "cross_paper.json", "w") as f:
        json.dump(cross_paper, f, indent=2)
    
    logger.info(f"Evaluation complete! Results saved to {CONFIG['output_dir']}")
    logger.info(f"Total triples evaluated: {len(all_judgments)}")
    logger.info(f"Overall relation precision: {summary['overall_relation_precision']}%")


# ==============================================================================
# JUDGE VALIDATION
# ==============================================================================

def validate_judge():
    """Validate judge accuracy by manually checking 20 random judgments."""
    logger.info("Starting judge validation...")
    
    # Load judgments
    judgments_path = CONFIG["output_dir"] / "judgments.csv"
    if not judgments_path.exists():
        logger.error("judgments.csv not found. Run evaluation first.")
        return
    
    judgments = []
    with open(judgments_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        judgments = list(reader)
    
    # Stratified sample: 10 supported, 10 unsupported
    supported = [j for j in judgments if j["is_supported"] == "True"]
    unsupported = [j for j in judgments if j["is_supported"] == "False"]
    
    sample_supported = random.sample(supported, min(10, len(supported)))
    sample_unsupported = random.sample(unsupported, min(10, len(unsupported)))
    
    validation_sample = sample_supported + sample_unsupported
    random.shuffle(validation_sample)
    
    logger.info(f"Validation sample: {len(sample_supported)} supported, {len(sample_unsupported)} unsupported")
    
    # For each sample, verify by checking the PDF text
    validation_results = []
    
    for j in validation_sample:
        paper_id = j["paper_id"]
        evidence = j["evidence"]
        
        # Get PDF text
        pdf_text = get_pdf_text(paper_id)
        
        # Manual verification: check if evidence is in PDF
        if pdf_text and evidence.lower() in pdf_text.lower():
            manual_supported = True
            manual_reason = "Evidence found in PDF"
        else:
            manual_supported = False
            manual_reason = "Evidence not found in PDF"
        
        # Compare with judge
        judge_supported = j["is_supported"] == "True"
        agreement = "AGREE" if manual_supported == judge_supported else "DISAGREE"
        
        validation_results.append({
            **j,
            "manual_verification": manual_reason,
            "agreement": agreement
        })
    
    # Save validation results
    with open(CONFIG["output_dir"] / "judge_validation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "source", "relation", "target", "is_supported",
            "judge_reason", "evidence", "confidence", "manual_verification", "agreement"
        ])
        writer.writeheader()
        for vr in validation_results:
            writer.writerow(vr)
    
    logger.info(f"Judge validation complete. Results saved to {CONFIG['output_dir']}/judge_validation.csv")
    logger.info("NOTE: Manual verification required. Update 'manual_verification' and 'agreement' columns.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate knowledge graph quality")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inspect command
    subparsers.add_parser("inspect", help="Inspect graph schema")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--sample", type=int, default=0,
                            help="Sample size per paper (0 = all triples)")
    
    # Validate command
    subparsers.add_parser("validate", help="Validate judge accuracy")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        inspect_schema()
    elif args.command == "evaluate":
        run_evaluation(args.sample)
    elif args.command == "validate":
        validate_judge()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
