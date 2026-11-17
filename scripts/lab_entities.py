"""
Phase 1b: Schema-free entity extraction from chunks.
Extractors: yake (default), spacy_noun_chunks, keybert (optional), llm_open (optional, off by default).
"""

import argparse
import csv
import json
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import requests

from lab_common import (
    get_config,
    print_section,
    print_warning,
    print_error
)


def verify_all_models():
    """Import and run model verification."""
    from lab_models import main as verify_models_main
    old_argv = sys.argv
    sys.argv = ["lab_models.py"]
    try:
        try:
            verify_models_main()
        except SystemExit as e:
            if e.code != 0:
                sys.exit(e.code)
    finally:
        sys.argv = old_argv


# Stopwords for filtering
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "here", "there", "then", "once", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "again", "further",
    "while", "until", "although", "though", "because", "since", "unless", "whether", "while",
    # Generic academic words to drop
    "results", "number", "data", "pages", "model", "network", "method", "approach", "paper",
    "figure", "table", "proposed", "study", "analysis", "based", "using", "show", "present",
    "provide", "consider", "however", "therefore", "thus", "hence", "within", "without"
}


def normalize_term(term: str) -> str:
    """Normalize term: lowercase, strip punctuation, lemmatize if possible."""
    term = term.lower().strip()
    term = term.strip(string.punctuation)
    # Remove citation markers like [1], (Smith et al., 2020)
    term = re.sub(r'\[\d+\]', '', term)
    term = re.sub(r'\([^)]*\d{4}[^)]*\)', '', term)
    term = term.strip()
    return term


def is_valid_term(term: str) -> bool:
    """Check if term is valid (not stopword, number, single letter, etc.)."""
    if not term or len(term) < 2:
        return False
    if term in STOPWORDS:
        return False
    if term.isdigit():
        return False
    if re.match(r'^[a-z]$', term):
        return False
    # Drop single letters and very short words
    if len(term) < 3:
        return False
    # Prefer multi-word phrases or acronyms, but allow single words >= 4 chars
    return True


class YAKEExtractor:
    """YAKE keyphrase extraction (statistical, no model download)."""
    
    def __init__(self, max_ngram: int = 3):
        try:
            import yake
            self.yake = yake
            self.max_ngram = max_ngram
        except ImportError:
            print_error("YAKE not installed. Run: uv pip install yake")
            raise
    
    def extract(self, text: str, max_terms: int = 8) -> List[str]:
        """Extract keyphrases from text."""
        kw_extractor = self.yake.KeywordExtractor(
            lan="en",
            n=self.max_ngram,
            dedupLim=0.7,
            top=max_terms
        )
        keywords = kw_extractor.extract_keywords(text)
        return [kw for kw, score in keywords]


class SpacyNounChunkExtractor:
    """spaCy noun phrase extraction (unsupervised)."""
    
    def __init__(self):
        try:
            import spacy
            self.spacy = spacy
            self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            print_warning("spaCy not available, falling back to YAKE")
            raise ImportError("spaCy not available")
        except OSError:
            print_warning("spaCy model not downloaded. Run: python -m spacy download en_core_web_sm")
            print_warning("Falling back to YAKE")
            raise ImportError("spaCy model not available")
    
    def extract(self, text: str, max_terms: int = 8) -> List[str]:
        """Extract noun phrases from text."""
        doc = self.nlp(text[:100000])  # Limit to avoid memory issues
        phrases = []
        for chunk in doc.noun_chunks:
            phrase = chunk.lemma_.lower().strip()
            if is_valid_term(phrase):
                phrases.append(phrase)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:max_terms]


class KeyBERTExtractor:
    """KeyBERT extraction (optional, requires sentence-transformers)."""
    
    def __init__(self):
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT()
        except ImportError:
            print_warning("KeyBERT not available. Install with: uv pip install keybert sentence-transformers")
            raise ImportError("KeyBERT not available")
    
    def extract(self, text: str, max_terms: int = 8) -> List[str]:
        """Extract keyphrases using KeyBERT."""
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top=max_terms
        )
        return [kw for kw, score in keywords]


class LLMOpenExtractor:
    """LLM-based extraction with open prompt (optional, uses tokens)."""
    
    def __init__(self, model_id: str, api_key: str, base_url: str):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
    
    def extract(self, text: str, max_terms: int = 8) -> List[str]:
        """Extract entities using LLM with open prompt."""
        prompt = f"List the key technical terms in this text, nothing else:\n\n{text[:2000]}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
            "X-Title": "Graph RAG Lab"
        }
        
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse terms (one per line or comma-separated)
        terms = []
        for line in content.strip().split("\n"):
            line = line.strip("-,.* ")
            if line and is_valid_term(line):
                terms.append(normalize_term(line))
        
        return terms[:max_terms]


def get_extractor(config):
    """Get the configured entity extractor."""
    if config.entity_extractor == "yake":
        return YAKEExtractor(max_ngram=config.ngram_max)
    elif config.entity_extractor == "spacy_noun_chunks":
        try:
            return SpacyNounChunkExtractor()
        except ImportError:
            print_warning("spaCy failed, falling back to YAKE")
            return YAKEExtractor(max_ngram=config.ngram_max)
    elif config.entity_extractor == "keybert":
        try:
            return KeyBERTExtractor()
        except ImportError:
            print_warning("KeyBERT not available, falling back to YAKE")
            return YAKEExtractor(max_ngram=config.ngram_max)
    elif config.entity_extractor == "llm_open":
        return LLMOpenExtractor(
            config.chat_model,
            config.openrouter_api_key,
            config.openrouter_base_url
        )
    else:
        print_warning(f"Unknown extractor {config.entity_extractor}, using YAKE")
        return YAKEExtractor(max_ngram=config.ngram_max)


def extract_entities_from_chunks(chunks_file: str, config) -> List[Dict]:
    """Extract entities from all chunks using TF-IDF ranking."""
    print_section("Phase 1b: Entity Extraction")
    
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print_error(f"Chunks file not found: {chunks_file}")
        sys.exit(1)
    
    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    print(f"Extractor: {config.entity_extractor}")
    print(f"Max entities per chunk: {config.max_entities_per_chunk}")
    
    extractor = get_extractor(config)
    
    # Step 1: Extract raw entities per chunk
    print("\nStep 1: Extracting raw entities...")
    chunk_raw_entities = []
    for chunk in chunks:
        text = chunk["text"]
        entities = extractor.extract(text, config.max_entities_per_chunk * 3)  # Get more candidates
        chunk_raw_entities.append(entities)
    
    print(f"  Extracted {sum(len(e) for e in chunk_raw_entities)} raw candidates")
    
    # Step 2: Filter and normalize
    print("\nStep 2: Filtering and normalizing...")
    chunk_filtered_entities = []
    for entities in chunk_raw_entities:
        filtered = []
        for term in entities:
            norm = normalize_term(term)
            if is_valid_term(norm):
                filtered.append(norm)
        chunk_filtered_entities.append(filtered)
    
    print(f"  After validity filter: {sum(len(e) for e in chunk_filtered_entities)} candidates")
    
    # Step 3: Compute TF-IDF scores
    print("\nStep 3: Computing TF-IDF scores...")
    from collections import defaultdict
    import math
    
    # Count term frequency per chunk
    chunk_term_freq = []
    for entities in chunk_filtered_entities:
        counter = Counter(entities)
        chunk_term_freq.append(counter)
    
    # Count document frequency (how many chunks contain each term)
    doc_freq = Counter()
    for counter in chunk_term_freq:
        for term in counter:
            doc_freq[term] += 1
    
    total_docs = len(chunks)
    
    # Compute TF-IDF and rank per chunk
    chunk_ranked_entities = []
    for i, counter in enumerate(chunk_term_freq):
        scored = []
        for term, tf in counter.items():
            idf = math.log(total_docs / (doc_freq[term] + 1))
            tfidf = tf * idf
            scored.append((term, tfidf))
        # Sort by TF-IDF descending
        scored.sort(key=lambda x: x[1], reverse=True)
        chunk_ranked_entities.append([t for t, s in scored])
    
    print(f"  Ranked entities by TF-IDF")
    
    # Step 4: Build paper-level entity counts (before min_paper_freq filter)
    print("\nStep 4: Building paper-level entity lists...")
    paper_entities = defaultdict(lambda: Counter())
    for chunk, entities in zip(chunks, chunk_ranked_entities):
        for term in entities[:config.max_entities_per_chunk]:
            paper_entities[chunk["paper_id"]][term] += 1
    
    # Step 5: Find terms in >50% of papers and drop them
    print("\nStep 5: Filtering overly common terms (>50% of papers)...")
    n_papers = len(paper_entities)
    term_paper_count = defaultdict(set)
    for paper_id, counter in paper_entities.items():
        for term in counter:
            term_paper_count[term].add(paper_id)
    
    common_terms = {term for term, papers in term_paper_count.items() 
                    if len(papers) > 0.5 * n_papers}
    print(f"  Dropping {len(common_terms)} terms appearing in >50% of papers")
    if len(common_terms) > 0:
        print(f"  Sample: {list(common_terms)[:10]}")
    
    # Step 6: Apply min_paper_freq filter at paper level only (for paper_entities.json)
    # Adaptive threshold: for small corpora, use lower threshold
    adaptive_min_freq = min(config.min_paper_freq, max(1, n_papers // 3))
    print(f"\nStep 6: Applying min_paper_freq filter for paper-level entities (adaptive: {adaptive_min_freq}, papers: {n_papers})...")
    paper_entities_filtered = {}
    for paper_id, counter in paper_entities.items():
        # First drop common terms
        filtered_counter = Counter({k: v for k, v in counter.items() if k not in common_terms})
        # Then apply adaptive min_paper_freq (across papers)
        filtered_counter = Counter({k: v for k, v in filtered_counter.items() 
                                     if len(term_paper_count[k]) >= adaptive_min_freq})
        # Limit to max_entities_per_paper
        filtered = filtered_counter.most_common(config.max_entities_per_paper)
        paper_entities_filtered[paper_id] = [
            {"term": term, "freq": freq, "n_chunks": freq}
            for term, freq in filtered
        ]
    
    # For chunks: use top TF-IDF terms directly (no shared term filter)
    # This ensures each chunk has its most relevant terms
    print("\nStep 7: Updating chunks with top TF-IDF entities (no shared term filter)...")
    for i, chunk in enumerate(chunks):
        ranked = chunk_ranked_entities[i]
        # Take top entities directly from TF-IDF ranking
        chunk["entities"] = ranked[:config.max_entities_per_chunk]
    
    # Stats
    entity_counts = [len(c["entities"]) for c in chunks]
    chunks_with_zero = sum(1 for c in entity_counts if c == 0)
    
    print(f"\nEntity statistics:")
    print(f"  Mean entities per chunk: {sum(entity_counts)/len(entity_counts):.2f}")
    print(f"  Median entities per chunk: {sorted(entity_counts)[len(entity_counts)//2]:.0f}")
    print(f"  Chunks with 0 entities: {chunks_with_zero}/{len(chunks)} ({100*chunks_with_zero/len(chunks):.1f}%)")
    print(f"  Max entities per chunk: {max(entity_counts)}")
    
    # Save updated chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # Save paper entities
    paper_entities_path = Path(config.data_dir) / "paper_entities.json"
    with open(paper_entities_path, "w", encoding="utf-8") as f:
        json.dump(paper_entities_filtered, f, indent=2)
    
    # Save shared terms (from all chunks, not just filtered paper entities)
    shared_terms = []
    for term, papers in term_paper_count.items():
        shared_terms.append({
            "term": term,
            "n_papers": len(papers),
            "papers": ",".join(sorted(papers))
        })
    shared_terms.sort(key=lambda x: x["n_papers"], reverse=True)
    
    shared_terms_path = Path(config.data_dir) / "shared_terms.csv"
    with open(shared_terms_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "n_papers", "papers"])
        writer.writeheader()
        writer.writerows(shared_terms)
    
    # Top 30 shared terms
    print("\nTop 30 shared terms:")
    for st in shared_terms[:30]:
        print(f"  {st['term']}: {st['n_papers']} papers")
    
    # Per-paper top terms
    print("\nTop 15 terms per paper:")
    for paper_id, entities in list(paper_entities_filtered.items())[:3]:
        print(f"\n  {paper_id}:")
        for ent in entities[:15]:
            print(f"    {ent['term']} (freq={ent['freq']})")
    
    # Random sample of chunks with entities
    print("\nRandom sample of 20 chunks with entities (text snippet + terms):")
    sample = random.sample(chunks, min(20, len(chunks)))
    for chunk in sample:
        print(f"  [{chunk['chunk_id']}]")
        print(f"    Text: {chunk['text'][:100]}...")
        print(f"    Terms: {chunk['entities'][:5]}")
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: Extract entities from chunks")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--skip", action="store_true", help="Skip entity extraction (entities stay empty)")
    args = parser.parse_args()
    
    config = get_config()
    
    if args.skip or not config.extract_entities:
        print("Entity extraction disabled (--skip or extract_entities=false)")
        print("Chunks will have empty entity lists")
        return
    
    # Verify models if using LLM extractor
    if config.entity_extractor == "llm_open":
        print_section("Verifying OpenRouter Models")
        verify_all_models()
    
    extract_entities_from_chunks(args.chunks, config)
    
    print_section("Done")


if __name__ == "__main__":
    main()
