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

# Download NLTK stopwords if needed
try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print_warning("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    print(f"Using NLTK stopwords ({len(STOPWORDS)} words)")
except ImportError:
    # Fallback to sklearn if NLTK not available
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOPWORDS = set(ENGLISH_STOP_WORDS)
        print_warning(f"NLTK not available. Using sklearn ENGLISH_STOP_WORDS as fallback ({len(STOPWORDS)} words)")
    except ImportError:
        # Last resort: empty set (YAKE has built-in stopword handling)
        print_warning("Neither NLTK nor sklearn available. Using YAKE's built-in stopword handling only.")
        STOPWORDS = set()


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
    
    raw_count = sum(len(e) for e in chunk_raw_entities)
    filtered_count = sum(len(e) for e in chunk_filtered_entities)
    print(f"  After validity filter (stopwords, length, etc.): {filtered_count} candidates")
    print(f"    Removed by validity filter: {raw_count - filtered_count} ({(raw_count - filtered_count)/raw_count*100:.1f}%)")
    
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
    
    # Step 5: Compute document frequency from ALL filtered entities (full cleaned text of each paper)
    print(f"\nStep 5: Computing document frequency from full cleaned text of each paper...")
    n_papers = len(paper_entities)
    term_paper_count_full = defaultdict(set)
    for chunk, entities in zip(chunks, chunk_filtered_entities):
        for term in entities:
            term_paper_count_full[term].add(chunk["paper_id"])
    
    # Print distribution of term occurrence across papers
    freq_dist = Counter()
    for term, papers in term_paper_count_full.items():
        freq_dist[len(papers)] += 1
    print(f"  Distribution of distinct candidate terms across papers:")
    for n_papers_occurrence in sorted(freq_dist.keys()):
        print(f"    {n_papers_occurrence} papers: {freq_dist[n_papers_occurrence]} terms")
    
    # Print 20 most widespread terms
    most_widespread = sorted(term_paper_count_full.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    print(f"\n  20 most widespread terms:")
    for term, papers in most_widespread:
        print(f"    {term}: {len(papers)} papers")
    
    # Test different thresholds
    for threshold in [4, 5, 6]:
        common_terms = {term for term, papers in term_paper_count_full.items() if len(papers) >= threshold}
        print(f"  Threshold {threshold}: would drop {len(common_terms)} terms")
        if threshold == 6 and len(common_terms) > 0:
            print(f"    Terms dropped: {list(common_terms)[:20]}")
    
    # Step 6: Filter overly common terms (data-driven filter)
    print(f"\nStep 6: Filtering overly common terms (appearing in >= {config.doc_freq_threshold} papers)...")
    common_terms = {term for term, papers in term_paper_count_full.items() 
                    if len(papers) >= config.doc_freq_threshold}
    print(f"  Dropping {len(common_terms)} terms appearing in >= {config.doc_freq_threshold} of {n_papers} papers")
    if len(common_terms) > 0:
        print(f"  Sample: {list(common_terms)[:10]}")
    
    # Count how many entities would be removed by doc-freq filter
    entities_before_doc_freq = sum(len(chunk_ranked_entities[i][:config.max_entities_per_chunk]) for i in range(len(chunks)))
    print(f"  Entities before doc-freq filter: {entities_before_doc_freq}")
    
    # Track most commonly dropped terms (by document frequency)
    dropped_terms_sorted = sorted(
        [(term, len(term_paper_count_full[term])) for term in common_terms],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Step 7: Apply min_paper_freq filter at paper level only (for paper_entities.json)
    # Adaptive threshold: for small corpora, use lower threshold
    adaptive_min_freq = min(config.min_paper_freq, max(1, n_papers // 3))
    print(f"\nStep 7: Applying min_paper_freq filter for paper-level entities (adaptive: {adaptive_min_freq}, papers: {n_papers})...")
    
    # Count entities before min_paper_freq filter
    entities_before_min_freq = sum(len(counter) for counter in paper_entities.values())
    
    paper_entities_filtered = {}
    for paper_id, counter in paper_entities.items():
        # First drop common terms
        filtered_counter = Counter({k: v for k, v in counter.items() if k not in common_terms})
        # Then apply adaptive min_paper_freq (across papers)
        filtered_counter = Counter({k: v for k, v in filtered_counter.items() 
                                     if len(term_paper_count_full[k]) >= adaptive_min_freq})
        # Limit to max_entities_per_paper
        filtered = filtered_counter.most_common(config.max_entities_per_paper)
        paper_entities_filtered[paper_id] = [
            {"term": term, "freq": freq, "n_chunks": freq}
            for term, freq in filtered
        ]
    
    entities_after_min_freq = sum(len(entities) for entities in paper_entities_filtered.values())
    print(f"  Paper-level entities before min_paper_freq: {entities_before_min_freq}")
    print(f"  Paper-level entities after min_paper_freq: {entities_after_min_freq}")
    print(f"    Removed by min_paper_freq filter: {entities_before_min_freq - entities_after_min_freq} ({(entities_before_min_freq - entities_after_min_freq)/entities_before_min_freq*100:.1f}%)")
    
    # For chunks: use top TF-IDF terms, but filter out common terms by document frequency
    print("\nStep 8: Updating chunks with top TF-IDF entities (with document-frequency filter)...")
    for i, chunk in enumerate(chunks):
        ranked = chunk_ranked_entities[i]
        # Filter out common terms by document frequency
        filtered_ranked = [t for t in ranked if t not in common_terms]
        # Take top entities from filtered list
        chunk["entities"] = filtered_ranked[:config.max_entities_per_chunk]
    
    entities_after_doc_freq = sum(len(c["entities"]) for c in chunks)
    print(f"  Entities after doc-freq filter: {entities_after_doc_freq}")
    print(f"    Removed by doc-freq filter: {entities_before_doc_freq - entities_after_doc_freq} ({(entities_before_doc_freq - entities_after_doc_freq)/entities_before_doc_freq*100:.1f}%)")
    
    # Stats
    entity_counts = [len(c["entities"]) for c in chunks]
    chunks_with_zero = sum(1 for c in entity_counts if c == 0)
    chunks_with_fewer_than_8 = sum(1 for c in entity_counts if c < 8)
    
    print(f"\nEntity statistics:")
    print(f"  Mean entities per chunk: {sum(entity_counts)/len(entity_counts):.2f}")
    print(f"  Median entities per chunk: {sorted(entity_counts)[len(entity_counts)//2]:.0f}")
    print(f"  Chunks with 0 entities: {chunks_with_zero}/{len(chunks)} ({100*chunks_with_zero/len(chunks):.1f}%)")
    print(f"  Chunks with fewer than 8 entities: {chunks_with_fewer_than_8}/{len(chunks)} ({100*chunks_with_fewer_than_8/len(chunks):.1f}%)")
    print(f"  Max entities per chunk: {max(entity_counts)}")
    
    # Print 10 random terms removed by min_paper_freq=2 but kept by 1
    print(f"\n10 random terms removed by min_paper_freq=2 but kept by 1:")
    terms_removed_by_2 = []
    for paper_id, counter in paper_entities.items():
        for term, freq in counter.items():
            if len(term_paper_count_full[term]) == 1 and freq >= 1:
                terms_removed_by_2.append(term)
    if len(terms_removed_by_2) > 0:
        sample_terms = random.sample(terms_removed_by_2, min(10, len(terms_removed_by_2)))
        for term in sample_terms:
            print(f"  {term}")
    else:
        print(f"  No such terms found")
    
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
    for term, papers in term_paper_count_full.items():
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
    
    # Top 10 dropped terms by document frequency
    print(f"\nTop 10 terms dropped by document-frequency filter (>= {config.doc_freq_threshold} papers):")
    for term, n_papers in dropped_terms_sorted[:10]:
        print(f"  {term}: {n_papers} papers")
    
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
        text_snippet = chunk['text'][:100].encode('ascii', 'ignore').decode('ascii')
        print(f"    Text: {text_snippet}...")
        print(f"    Terms: {chunk['entities'][:5]}")
    
    # Write detailed output to file for analysis
    output_path = Path(config.data_dir) / "entity_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("20 random chunks with entities:\n")
        f.write("=" * 80 + "\n\n")
        for chunk in sample:
            f.write(f"[{chunk['chunk_id']}]\n")
            f.write(f"Text: {chunk['text'][:200]}...\n")
            f.write(f"Terms: {chunk['entities']}\n\n")
        
        f.write("\nTop 10 terms per paper:\n")
        f.write("=" * 80 + "\n\n")
        for paper_id, entities in paper_entities_filtered.items():
            f.write(f"{paper_id}:\n")
            for ent in entities[:10]:
                f.write(f"  {ent['term']} (freq={ent['freq']})\n")
            f.write("\n")
        
        # Analyze junk in entities
        f.write("\nJunk analysis:\n")
        f.write("=" * 80 + "\n\n")
        all_entities = []
        for chunk in chunks:
            all_entities.extend(chunk['entities'])
        
        import re
        entities_with_digits = [e for e in all_entities if re.search(r'\d', e)]
        entities_with_urls = [e for e in all_entities if re.search(r'http|www|\.com|\.org|\.edu', e, re.IGNORECASE)]
        entities_single_char = [e for e in all_entities if len(e.strip()) == 1]
        entities_with_et_al = [e for e in all_entities if 'et al' in e.lower()]
        entities_with_fig = [e for e in all_entities if re.search(r'\bfig\b', e, re.IGNORECASE)]
        
        f.write(f"Total entities: {len(all_entities)}\n")
        f.write(f"Entities containing digits: {len(entities_with_digits)} ({len(entities_with_digits)/len(all_entities)*100:.1f}%)\n")
        f.write(f"Entities with URLs: {len(entities_with_urls)} ({len(entities_with_urls)/len(all_entities)*100:.1f}%)\n")
        f.write(f"Single character entities: {len(entities_single_char)} ({len(entities_single_char)/len(all_entities)*100:.1f}%)\n")
        f.write(f"Entities with 'et al': {len(entities_with_et_al)} ({len(entities_with_et_al)/len(all_entities)*100:.1f}%)\n")
        f.write(f"Entities with 'fig': {len(entities_with_fig)} ({len(entities_with_fig)/len(all_entities)*100:.1f}%)\n")
        
        if entities_with_digits:
            f.write(f"\nSample entities with digits (10): {entities_with_digits[:10]}\n")
        if entities_with_urls:
            f.write(f"\nSample entities with URLs: {entities_with_urls}\n")
        if entities_single_char:
            f.write(f"\nSingle character entities: {entities_single_char}\n")
        if entities_with_et_al:
            f.write(f"\nSample entities with 'et al' (10): {entities_with_et_al[:10]}\n")
        if entities_with_fig:
            f.write(f"\nSample entities with 'fig' (10): {entities_with_fig[:10]}\n")
    
    print(f"\nDetailed analysis written to: {output_path}")
    
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
