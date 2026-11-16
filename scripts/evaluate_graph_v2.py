"""
Improved knowledge graph evaluation with three distinct models.
"""

import sys
import json
import csv
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import logging

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
    logger = logging.getLogger(__name__)
    logger.warning("rapidfuzz not installed. Quote verification will use exact matching only.")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo4j import GraphDatabase
from graph_rag.config import settings
from graph_rag.llm.model_client import CloudModelClient

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration
CONFIG = {
    "dataset_id": settings.DATASET_ID,
    "output_dir": PROJECT_ROOT / "eval_results",
    "entity_label": "Entity",
    "paper_label": "Paper",
    "chunk_label": "Chunk",
    "mention_label": "Mention",
    
    # Entity properties
    "entity_name_prop": "name",
    "entity_aliases_prop": "aliases",
    "entity_category_prop": "category",
    "entity_id_prop": "entity_id",
    
    # Relationship properties
    "relation_evidence_prop": "evidence",
    "relation_confidence_prop": "confidence",
    "relation_chunk_id_prop": "chunk_id",
    "relation_paper_id_prop": "paper_id",
    
    # Semantic relationships to EVALUATE
    "semantic_rels": {
        "RELATED_TO", "USES_METHOD", "EVALUATES_ON", "REPORTS",
        "PROPOSES", "AUTHORED", "AFFILIATED_WITH", "PART_OF_TOPIC"
    },
    
    # Paper-to-Entity path
    "paper_to_entity_path": "MENTIONS",
    
    # Models - will be validated
    "extractor_model": settings.EXTRACTION_MODEL,
    "judge_model": settings.EVALUATION_MODEL,
    "audit_model": settings.AUDIT_MODEL,
    
    "judge_temperature": 0.0,
    "audit_temperature": 0.0,
    
    # Audit sample size
    "audit_sample_size": 40,
    "audit_seed": 42,
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# Create model clients
judge_client = CloudModelClient(llm_model=CONFIG["judge_model"], embed_model="")
audit_client = CloudModelClient(llm_model=CONFIG["audit_model"], embed_model="")


def validate_models():
    """Ensure extractor and judge are distinct. Audit can be same as judge if needed."""
    models = [CONFIG["extractor_model"], CONFIG["judge_model"]]
    if len(set(models)) != 2:
        raise ValueError(
            f"Extractor and Judge must be distinct. Got: Extractor={CONFIG['extractor_model']}, "
            f"Judge={CONFIG['judge_model']}"
        )
    
    # Check if audit is distinct
    if CONFIG["audit_model"] == CONFIG["extractor_model"]:
        logger.warning("Audit model same as extractor - not ideal but proceeding")
    elif CONFIG["audit_model"] == CONFIG["judge_model"]:
        logger.warning("Audit model same as judge - noted as limitation in report")
    else:
        logger.info("All three models are distinct")
    
    logger.info(f"Models: Extractor={CONFIG['extractor_model']}, "
                f"Judge={CONFIG['judge_model']}, Audit={CONFIG['audit_model']}")


def get_driver():
    """Create Neo4j driver."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        connection_timeout=30.0,
    )


def get_papers(driver) -> List[Dict[str, Any]]:
    """Get all papers in the dataset."""
    with driver.session() as session:
        # First try with dataset_id filter
        result = session.run("""
            MATCH (p:Paper {dataset_id: $dataset_id})
            RETURN p.uid as uid, p.paper_id as paper_id, p.title as title
            ORDER BY p.paper_id
        """, dataset_id=CONFIG["dataset_id"])
        papers = [dict(record) for record in result]
        
        logger.info(f"Found {len(papers)} papers with dataset_id={CONFIG['dataset_id']}")
        for p in papers:
            logger.info(f"  - {p['paper_id']}")
        
        # If no papers found, try without dataset_id filter
        if len(papers) == 0:
            logger.warning(f"No papers found with dataset_id={CONFIG['dataset_id']}, trying without filter")
            result = session.run("""
                MATCH (p:Paper)
                RETURN p.uid as uid, p.paper_id as paper_id, p.title as title, p.dataset_id as dataset_id
                ORDER BY p.paper_id
            """)
            papers = [dict(record) for record in result]
            logger.info(f"Found {len(papers)} papers without dataset_id filter:")
            for p in papers:
                logger.info(f"  - {p['paper_id']} (dataset_id: {p.get('dataset_id', 'N/A')})")
        
        return papers


def get_entities_for_paper(driver, paper_uid: str, paper_dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all entities for a paper."""
    dataset_id = paper_dataset_id or CONFIG["dataset_id"]
    with driver.session() as session:
        # Try multiple relationship paths
        result = session.run("""
            MATCH (p:Paper {uid: $paper_uid})
            OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e2:Entity)
            WITH COALESCE(e, e2) as e
            WHERE e IS NOT NULL
            RETURN DISTINCT e.uid as uid, e.entity_id as entity_id, e.name as name, 
                   e.category as category, e.aliases as aliases
            ORDER BY e.name
        """, paper_uid=paper_uid)
        entities = [dict(record) for record in result]
        logger.info(f"Found {len(entities)} entities for paper {paper_uid}")
        return entities


def get_semantic_triples_for_paper(driver, paper_id: str) -> List[Dict[str, Any]]:
    """Get all semantic relationships (triples) for a paper."""
    semantic_rels_list = list(CONFIG["semantic_rels"])
    
    with driver.session() as session:
        query = f"""
            MATCH (s:Entity {{dataset_id: $dataset_id}})-[r:{'|'.join(semantic_rels_list)}]->(t:Entity {{dataset_id: $dataset_id}})
            WHERE r.dataset_id = $dataset_id AND r.paper_id = $paper_id
            RETURN s.name as source, type(r) as relation, t.name as target,
                   s.category as source_category, t.category as target_category,
                   s.aliases as source_aliases, t.aliases as target_aliases,
                   r.evidence as evidence, r.confidence as confidence,
                   r.chunk_id as chunk_id, r.paper_id as paper_id,
                   s.entity_id as source_entity_id, t.entity_id as target_entity_id
        """
        result = session.run(query, dataset_id=CONFIG["dataset_id"], paper_id=paper_id)
        return [dict(record) for record in result]


def get_pdf_text(paper_id: str) -> str:
    """Extract full text from PDF."""
    pdf_path = settings.PDF_DIR / paper_id
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return ""
    
    try:
        import pymupdf  # fitz is deprecated
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return ""


def get_chunk_text(driver, chunk_id: str) -> str:
    """Get text for a chunk."""
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk {uid: $chunk_id, dataset_id: $dataset_id})
            RETURN c.text as text
        """, chunk_id=chunk_id, dataset_id=CONFIG["dataset_id"])
        record = result.single()
        return record["text"] if record else ""


def get_neighbor_chunks(driver, chunk_id: str, paper_id: str) -> List[str]:
    """Get neighboring chunks (before and after)."""
    # Simplified: just return empty list since page_number doesn't exist
    return []


def find_top_passages(pdf_text: str, query: str, top_k: int = 3) -> List[str]:
    """Find top-k passages by simple overlap (no embeddings for now)."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', pdf_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Score by overlap with query words
    query_words = set(query.lower().split())
    scored = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(query_words & sent_words)
        if overlap > 0:
            scored.append((overlap, sent))
    
    # Sort by overlap and return top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sent for _, sent in scored[:top_k]]


def judge_triple(triple: Dict[str, Any], pdf_text: str, chunks: List[str]) -> Dict[str, Any]:
    """Judge a triple using the judge model."""
    source = triple["source"]
    target = triple["target"]
    relation = triple["relation"]
    evidence = triple.get("evidence", "")
    source_aliases = triple.get("source_aliases", [])
    target_aliases = triple.get("target_aliases", [])
    
    # Build context
    source_text = f"{source} (aliases: {', '.join(source_aliases) if source_aliases else 'none'})"
    target_text = f"{target} (aliases: {', '.join(target_aliases) if target_aliases else 'none'})"
    
    # Get top passages
    query = f"{source} {relation} {target}"
    top_passages = find_top_passages(pdf_text, query, top_k=3)
    
    # Build prompt
    prompt = f"""You are evaluating whether a knowledge graph triple is supported by a research paper.

Triple to evaluate:
- Head entity: {source_text}
- Relation: {relation}
- Tail entity: {target_text}

Evidence from extraction: "{evidence}"

Neighboring text chunks:
{chr(10).join(f"- {c[:500]}" for c in chunks[:3])}

Top relevant passages from paper:
{chr(10).join(f"- {p}" for p in top_passages)}

Instructions:
1. Coreference resolution: "they", "we", "this method", and aliases count as mentions of the entity.
2. Support levels:
   - "explicit": The paper directly states the relationship with clear language.
   - "implied": The relationship can be reasonably inferred from context, though not explicitly stated.
   - "unsupported": No evidence supports this relationship.
3. Evaluate head_valid and tail_valid: Are the entities actually mentioned in the text (including via aliases/coreference)?
4. Evaluate relation_correct: Is this the correct relation type?
5. Evaluate direction_correct: Is the direction correct (head -> tail)?

Return JSON only:
{{
  "head_valid": true/false,
  "tail_valid": true/false,
  "support": "explicit"|"implied"|"unsupported",
  "relation_correct": true/false,
  "direction_correct": true/false,
  "reason": "brief explanation"
}}"""

    # Call LLM with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = judge_client.generate(
                prompt=prompt,
                temperature=CONFIG["judge_temperature"],
                response_format={"type": "json_object"}
            )
            
            # Parse JSON
            result = json.loads(response)
            
            # Validate structure
            required_keys = ["head_valid", "tail_valid", "support", "relation_correct", "direction_correct", "reason"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing key: {key}")
            
            if result["support"] not in ["explicit", "implied", "unsupported"]:
                raise ValueError(f"Invalid support value: {result['support']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Judge attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                # Return default on failure
                return {
                    "head_valid": False,
                    "tail_valid": False,
                    "support": "unsupported",
                    "relation_correct": False,
                    "direction_correct": False,
                    "reason": f"Judge error: {str(e)}"
                }


def audit_triple(triple: Dict[str, Any], pdf_text: str) -> Dict[str, Any]:
    """Audit a triple using the audit model with full paper text."""
    source = triple["source"]
    target = triple["target"]
    relation = triple["relation"]
    source_aliases = triple.get("source_aliases", [])
    target_aliases = triple.get("target_aliases", [])
    
    source_text = f"{source} (aliases: {', '.join(source_aliases) if source_aliases else 'none'})"
    target_text = f"{target} (aliases: {', '.join(target_aliases) if target_aliases else 'none'})"
    
    # Log paper text length
    logger.info(f"Paper text length: {len(pdf_text)} chars, sending full text to auditor")
    
    prompt = f"""You are auditing a knowledge graph triple against the FULL text of a research paper.

Triple to audit:
- Head entity: {source_text}
- Relation: {relation}
- Tail entity: {target_text}

Full paper text:
{pdf_text}

Instructions:
1. Find the EXACT sentence(s) in the paper that support this triple.
2. Quote the sentence(s) verbatim.
3. If no support exists, quote should be empty.
4. Evaluate verdict, relation_correct, and direction_correct.

Return JSON only:
{{
  "verdict": "explicit"|"implied"|"unsupported",
  "relation_correct": true/false,
  "direction_correct": true/false,
  "quote": "exact sentence(s) from paper or empty string",
  "reason": "brief explanation"
}}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = audit_client.generate(
                prompt=prompt,
                temperature=CONFIG["audit_temperature"],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            
            required_keys = ["verdict", "relation_correct", "direction_correct", "quote", "reason"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing key: {key}")
            
            if result["verdict"] not in ["explicit", "implied", "unsupported"]:
                raise ValueError(f"Invalid verdict value: {result['verdict']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Audit attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "verdict": "unsupported",
                    "relation_correct": False,
                    "direction_correct": False,
                    "quote": "",
                    "reason": f"Audit error: {str(e)}"
                }


def verify_quote_in_paper(quote: str, pdf_text: str) -> bool:
    """Verify that a quote appears in the paper with robust normalization and fuzzy matching."""
    if not quote or not quote.strip():
        return True  # Empty quote is acceptable for unsupported
    
    # Normalize both strings
    def normalize_text(text: str) -> str:
        # NFKC normalization
        text = unicodedata.normalize("NFKC", text)
        # Join hyphen+newline
        text = re.sub(r'-\n', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        # Casefold for case-insensitive matching
        text = text.casefold()
        return text
    
    normalized_quote = normalize_text(quote)
    normalized_pdf = normalize_text(pdf_text)
    
    # Try exact substring match first
    if normalized_quote in normalized_pdf:
        return True
    
    # Try fuzzy matching if rapidfuzz is available
    if fuzz is not None:
        # Check partial ratio for fuzzy match
        ratio = fuzz.partial_ratio(normalized_quote, normalized_pdf)
        if ratio >= 90:
            return True
    
    return False


def compute_metrics(judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics from judgments."""
    total = len(judgments)
    if total == 0:
        return {}
    
    # Count by support
    explicit = sum(1 for j in judgments if j["support"] == "explicit")
    implied = sum(1 for j in judgments if j["support"] == "implied")
    unsupported = sum(1 for j in judgments if j["support"] == "unsupported")
    
    # Source Support % = (explicit + implied) / judged
    source_support_pct = round((explicit + implied) / total * 100, 2)
    
    # Relation Precision % = (support is explicit or implied AND relation_correct AND direction_correct) / judged
    relation_correct = sum(1 for j in judgments 
                          if j["support"] in ["explicit", "implied"] 
                          and j["relation_correct"] 
                          and j["direction_correct"])
    relation_precision_pct = round(relation_correct / total * 100, 2)
    
    # Strict Precision % = explicit only with correct relation and direction
    strict_correct = sum(1 for j in judgments 
                        if j["support"] == "explicit" 
                        and j["relation_correct"] 
                        and j["direction_correct"])
    strict_precision_pct = round(strict_correct / total * 100, 2)
    
    # Entity Precision % = unique entities marked valid / unique entities judged
    valid_entities = set()
    judged_entities = set()
    
    for j in judgments:
        judged_entities.add(j["source"])
        judged_entities.add(j["target"])
        if j["head_valid"]:
            valid_entities.add(j["source"])
        if j["tail_valid"]:
            valid_entities.add(j["target"])
    
    entity_precision_pct = round(len(valid_entities) / len(judged_entities) * 100, 2) if judged_entities else 0
    
    return {
        "total_judged": total,
        "explicit_count": explicit,
        "implied_count": implied,
        "unsupported_count": unsupported,
        "source_support_pct": source_support_pct,
        "relation_precision_pct": relation_precision_pct,
        "strict_precision_pct": strict_precision_pct,
        "entity_precision_pct": entity_precision_pct,
        "valid_entities": len(valid_entities),
        "judged_entities": len(judged_entities)
    }


def deduplicate_triples(triples: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Remove duplicate (head, relation, tail) triples and return deduplicated list + count removed."""
    seen = set()
    deduped = []
    for t in triples:
        key = (t["source"], t["relation"], t["target"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    removed = len(triples) - len(deduped)
    return deduped, removed


def run_evaluation(sample_size: int = 0):
    """Run the full evaluation."""
    logger.info("Starting evaluation...")
    validate_models()
    
    driver = get_driver()
    
    # Get papers
    papers = get_papers(driver)
    logger.info(f"Found {len(papers)} papers")
    
    # Store all data
    all_triples_by_paper = {}
    all_entities_by_paper = {}
    all_judgments = []
    total_duplicates_removed = 0
    
    for paper in papers:
        paper_uid = paper["uid"]
        paper_id = paper["paper_id"]
        
        logger.info(f"Processing paper: {paper_id}")
        
        # Get entities
        entities = get_entities_for_paper(driver, paper_uid)
        all_entities_by_paper[paper_id] = entities
        
        # Get semantic triples
        triples = get_semantic_triples_for_paper(driver, paper_id)
        
        # Deduplicate triples
        original_count = len(triples)
        triples, removed = deduplicate_triples(triples)
        total_duplicates_removed += removed
        if removed > 0:
            logger.info(f"Removed {removed} duplicate triples from {paper_id} ({original_count} -> {len(triples)})")
        
        # Sample if needed
        if sample_size > 0 and len(triples) > sample_size:
            triples = random.sample(triples, sample_size)
        
        all_triples_by_paper[paper_id] = triples
        
        # Get PDF text
        pdf_text = get_pdf_text(paper_id)
        
        # Judge each triple
        for i, triple in enumerate(triples):
            logger.info(f"Judging triple {i+1}/{len(triples)}")
            
            # Get chunk text and neighbors
            chunk_text = get_chunk_text(driver, triple["chunk_id"]) if triple.get("chunk_id") else ""
            neighbor_chunks = get_neighbor_chunks(driver, triple["chunk_id"], paper_id) if triple.get("chunk_id") else []
            chunks = [chunk_text] + neighbor_chunks
            
            # Judge
            judgment = judge_triple(triple, pdf_text, chunks)
            
            # Add metadata
            judgment["paper_id"] = paper_id
            judgment["source"] = triple["source"]
            judgment["relation"] = triple["relation"]
            judgment["target"] = triple["target"]
            judgment["evidence"] = triple.get("evidence", "")
            judgment["confidence"] = triple.get("confidence", 0)
            
            all_judgments.append(judgment)
    
    driver.close()
    
    # Compute per-paper metrics
    per_paper_results = {}
    for paper_id, triples in all_triples_by_paper.items():
        paper_judgments = [j for j in all_judgments if j["paper_id"] == paper_id]
        metrics = compute_metrics(paper_judgments)
        
        # Compute entity grounding, duplicate rate, orphan rate
        entities = all_entities_by_paper[paper_id]
        pdf_text = get_pdf_text(paper_id)
        
        # Entity grounding
        grounded = 0
        for e in entities:
            name = e["name"].lower()
            aliases = [a.lower() for a in e.get("aliases", [])]
            if name in pdf_text.lower() or any(a in pdf_text.lower() for a in aliases):
                grounded += 1
        
        entity_grounding_pct = round(grounded / len(entities) * 100, 2) if entities else 0
        
        # Duplicate rate (normalized names)
        normalized_names = {}
        for e in entities:
            norm = re.sub(r'[^\w]', '', e["name"].lower())
            if norm in normalized_names:
                normalized_names[norm] += 1
            else:
                normalized_names[norm] = 1
        duplicates = sum(1 for c in normalized_names.values() if c > 1)
        duplicate_rate_pct = round(duplicates / len(entities) * 100, 2) if entities else 0
        
        # Orphan rate (entities with no semantic relationships)
        entity_ids_in_rels = set()
        for t in triples:
            entity_ids_in_rels.add(t["source_entity_id"])
            entity_ids_in_rels.add(t["target_entity_id"])
        orphans = sum(1 for e in entities if e["entity_id"] not in entity_ids_in_rels)
        orphan_rate_pct = round(orphans / len(entities) * 100, 2) if entities else 0
        
        per_paper_results[paper_id] = {
            "paper_id": paper_id,
            "title": next(p["title"] for p in papers if p["paper_id"] == paper_id),
            "entity_count": len(entities),
            "triple_count": len(triples),
            "entity_grounding_pct": entity_grounding_pct,
            "duplicate_rate_pct": duplicate_rate_pct,
            "orphan_rate_pct": orphan_rate_pct,
            **metrics
        }
    
    # Compute relation type precision
    relation_type_counts = defaultdict(lambda: {"total": 0, "supported": 0, "correct": 0})
    for j in all_judgments:
        rel_type = j["relation"]
        relation_type_counts[rel_type]["total"] += 1
        if j["support"] in ["explicit", "implied"]:
            relation_type_counts[rel_type]["supported"] += 1
        if j["support"] in ["explicit", "implied"] and j["relation_correct"] and j["direction_correct"]:
            relation_type_counts[rel_type]["correct"] += 1
    
    relation_type_precision = {}
    for rel_type, counts in relation_type_counts.items():
        relation_type_precision[rel_type] = round(counts["correct"] / counts["total"] * 100, 2) if counts["total"] > 0 else 0
    
    # Overall metrics
    overall_metrics = compute_metrics(all_judgments)
    
    # Save results
    summary = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_id": CONFIG["dataset_id"],
        "sample_size_per_paper": sample_size,
        "total_papers": len(papers),
        "total_entities": sum(len(e) for e in all_entities_by_paper.values()),
        "total_triples": sum(len(t) for t in all_triples_by_paper.values()),
        "total_judgments": len(all_judgments),
        "extractor_model": CONFIG["extractor_model"],
        "judge_model": CONFIG["judge_model"],
        "audit_model": CONFIG["audit_model"],
        "overall_metrics": overall_metrics,
        "per_paper_results": per_paper_results,
        "relation_type_counts": dict(relation_type_counts),
        "relation_type_precision": relation_type_precision
    }
    
    with open(CONFIG["output_dir"] / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save judgments
    with open(CONFIG["output_dir"] / "judgments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "source", "relation", "target", "head_valid", "tail_valid",
            "support", "relation_correct", "direction_correct", "reason", "evidence", "confidence"
        ])
        writer.writeheader()
        for j in all_judgments:
            writer.writerow(j)
    
    logger.info(f"Evaluation complete. Results saved to {CONFIG['output_dir']}")
    if overall_metrics:
        logger.info(f"Overall relation precision: {overall_metrics.get('relation_precision_pct', 0)}%")
    else:
        logger.info("No triples to evaluate")
    logger.info(f"Total duplicate triples removed: {total_duplicates_removed}")
    
    # Add duplicate info to summary
    summary["total_duplicates_removed"] = total_duplicates_removed
    
    return summary


def run_audit(reference_file: Optional[str] = None):
    """Run independent audit on sampled triples."""
    logger.info("Starting independent audit...")
    validate_models()
    
    # Load judgments
    judgments_path = CONFIG["output_dir"] / "judgments.csv"
    if not judgments_path.exists():
        logger.error("judgments.csv not found. Run evaluation first.")
        return
    
    judgments = []
    with open(judgments_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        judgments = list(reader)
    
    # Deduplicate judgments before sampling
    original_judgment_count = len(judgments)
    judgments, removed = deduplicate_triples(judgments)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate judgments ({original_judgment_count} -> {len(judgments)})")
    
    # Load reference file if provided
    reference_data = {}
    if reference_file and Path(reference_file).exists():
        with open(reference_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["head"], row["relation"], row["tail"])
                reference_data[key] = row
        logger.info(f"Loaded {len(reference_data)} reference judgments from {reference_file}")
    
    # Stratified sample: 40 triples across papers and relation types
    random.seed(CONFIG["audit_seed"])
    
    # Group by paper and relation type
    groups = defaultdict(list)
    for j in judgments:
        key = (j["paper_id"], j["relation"])
        groups[key].append(j)
    
    # Sample from each group
    audit_sample = []
    for key, group_judgments in groups.items():
        sample_size = max(1, CONFIG["audit_sample_size"] // len(groups))
        if len(group_judgments) >= sample_size:
            audit_sample.extend(random.sample(group_judgments, sample_size))
        else:
            audit_sample.extend(group_judgments)
    
    # Trim to exact sample size
    if len(audit_sample) > CONFIG["audit_sample_size"]:
        audit_sample = random.sample(audit_sample, CONFIG["audit_sample_size"])
    
    logger.info(f"Audit sample size: {len(audit_sample)}")
    
    # Audit each triple
    audit_results = []
    driver = get_driver()
    
    for i, triple in enumerate(audit_sample):
        logger.info(f"Auditing triple {i+1}/{len(audit_sample)}")
        
        paper_id = triple["paper_id"]
        pdf_text = get_pdf_text(paper_id)
        
        # Get entity info
        entities = get_entities_for_paper(driver, paper_id)
        entity_map = {e["name"]: e for e in entities}
        
        # Add aliases to triple
        source_entity = entity_map.get(triple["source"], {})
        target_entity = entity_map.get(triple["target"], {})
        triple["source_aliases"] = source_entity.get("aliases", [])
        triple["target_aliases"] = target_entity.get("aliases", [])
        
        # Audit
        audit_result = audit_triple(triple, pdf_text)
        
        # Verify quote (do NOT downgrade verdict)
        quote_verified = verify_quote_in_paper(audit_result["quote"], pdf_text)
        
        # Compare with judge
        judge_supported = triple["support"] in ["explicit", "implied"]
        audit_supported = audit_result["verdict"] in ["explicit", "implied"]
        agree = "AGREE" if judge_supported == audit_supported else "DISAGREE"
        
        audit_results.append({
            "paper_id": paper_id,
            "head": triple["source"],
            "relation": triple["relation"],
            "tail": triple["target"],
            "judge_verdict": triple["support"],
            "audit_verdict": audit_result["verdict"],
            "quote": audit_result["quote"],
            "quote_verified": quote_verified,
            "agree": agree,
            "audit_relation_correct": audit_result["relation_correct"],
            "audit_direction_correct": audit_result["direction_correct"],
            "audit_reason": audit_result["reason"]
        })
    
    driver.close()
    
    # Save audit results
    with open(CONFIG["output_dir"] / "audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "head", "relation", "tail", "judge_verdict", "audit_verdict",
            "quote", "quote_verified", "agree", "audit_relation_correct",
            "audit_direction_correct", "audit_reason"
        ])
        writer.writeheader()
        for ar in audit_results:
            writer.writerow(ar)
    
    # Compute agreement metrics
    agree_count = sum(1 for ar in audit_results if ar["agree"] == "AGREE")
    agreement_pct = round(agree_count / len(audit_results) * 100, 2)
    
    # Cohen's kappa
    judge_supported = [1 if ar["judge_verdict"] in ["explicit", "implied"] else 0 for ar in audit_results]
    audit_supported = [1 if ar["audit_verdict"] in ["explicit", "implied"] else 0 for ar in audit_results]
    
    # Simple kappa calculation
    observed_agreement = agree_count / len(audit_results)
    
    # Expected agreement by chance
    judge_yes = sum(judge_supported) / len(judge_supported)
    audit_yes = sum(audit_supported) / len(audit_supported)
    expected_agreement = (judge_yes * audit_yes) + ((1 - judge_yes) * (1 - audit_yes))
    
    kappa = round((observed_agreement - expected_agreement) / (1 - expected_agreement), 3) if expected_agreement < 1 else 1.0
    
    # Auditor precision (with and without quote verification)
    audit_correct = sum(1 for ar in audit_results 
                       if ar["audit_verdict"] in ["explicit", "implied"] 
                       and ar["audit_relation_correct"] 
                       and ar["audit_direction_correct"])
    audit_precision_pct = round(audit_correct / len(audit_results) * 100, 2)
    
    # Precision with verified quotes only
    verified_quotes = sum(1 for ar in audit_results if ar["quote_verified"])
    audit_correct_verified = sum(1 for ar in audit_results 
                                if ar["audit_verdict"] in ["explicit", "implied"] 
                                and ar["audit_relation_correct"] 
                                and ar["audit_direction_correct"]
                                and ar["quote_verified"])
    audit_precision_verified_pct = round(audit_correct_verified / verified_quotes * 100, 2) if verified_quotes > 0 else 0
    
    # Compare with reference if provided
    reference_agreement = None
    if reference_data:
        ref_agree_count = 0
        ref_total = 0
        for ar in audit_results:
            key = (ar["head"], ar["relation"], ar["tail"])
            if key in reference_data:
                ref_total += 1
                ref_verdict = reference_data[key]["verdict"]
                audit_supported = ar["audit_verdict"] in ["explicit", "implied"]
                ref_supported = ref_verdict in ["explicit", "implied"]
                if audit_supported == ref_supported:
                    ref_agree_count += 1
        if ref_total > 0:
            reference_agreement = round(ref_agree_count / ref_total * 100, 2)
            logger.info(f"Reference agreement: {reference_agreement}% ({ref_agree_count}/{ref_total} samples)")
    
    logger.info(f"Audit complete. Agreement: {agreement_pct}%, Kappa: {kappa}")
    logger.info(f"Audit precision: {audit_precision_pct}% (all), {audit_precision_verified_pct}% (verified quotes only)")
    logger.info(f"Quote verification rate: {round(verified_quotes / len(audit_results) * 100, 2)}% ({verified_quotes}/{len(audit_results)})")
    
    return {
        "agreement_pct": agreement_pct,
        "kappa": kappa,
        "audit_precision_pct": audit_precision_pct,
        "audit_precision_verified_pct": audit_precision_verified_pct,
        "quote_verification_rate": round(verified_quotes / len(audit_results) * 100, 2) if audit_results else 0,
        "reference_agreement": reference_agreement,
        "audit_sample_size": len(audit_results)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate knowledge graph")
    parser.add_argument("command", choices=["evaluate", "audit"], help="Command to run")
    parser.add_argument("--sample", type=int, default=0, help="Sample size per paper (0 for all)")
    parser.add_argument("--reference", type=str, default=None, help="Reference CSV file for audit comparison")
    args = parser.parse_args()
    
    if args.command == "evaluate":
        run_evaluation(args.sample)
    elif args.command == "audit":
        run_audit(args.reference)


if __name__ == "__main__":
    main()
