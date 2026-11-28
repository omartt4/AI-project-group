"""
Test Chunk Quality Metrics.

Tests chunk length distribution, overlap, and information density.
"""

import statistics
from typing import List

def analyze_chunks(chunks: List[str]) -> dict:
    if not chunks:
        return {"error": "No chunks provided"}
    
    lengths = [len(chunk) for chunk in chunks]
    avg_length = statistics.mean(lengths)
    std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    min_length = min(lengths)
    max_length = max(lengths)
    
    # Overlap analysis
    overlaps = []
    for i in range(len(chunks) - 1):
        max_overlap = 0
        min_len = min(len(chunks[i]), len(chunks[i + 1]))
        for j in range(1, min_len + 1):
            if chunks[i][-j:] == chunks[i + 1][:j]:
                max_overlap = j
        overlaps.append(max_overlap)
    
    avg_overlap = statistics.mean(overlaps) if overlaps else 0.0
    
    # Information density
    densities = []
    for chunk in chunks:
        words = chunk.split()
        if words:
            unique_words = set(word.lower() for word in words)
            densities.append(len(unique_words) / len(words))
    
    avg_density = statistics.mean(densities) if densities else 0.0
    
    return {
        "total_chunks": len(chunks),
        "avg_length": avg_length,
        "std_length": std_length,
        "min_length": min_length,
        "max_length": max_length,
        "avg_overlap": avg_overlap,
        "avg_density": avg_density,
    }

# Test with sample chunks
print("Chunk Quality Metrics Test")
print("=" * 60)

sample_chunks = [
    "This is the first chunk of text. It contains some information about the topic.",
    "the topic. This is the second chunk that overlaps slightly with the first one.",
    "first one. This is the third chunk with different content and information.",
    "information. This is the fourth chunk that continues the discussion.",
]

print("\nSample Chunks:")
for i, chunk in enumerate(sample_chunks, 1):
    print(f"  Chunk {i}: {chunk[:50]}...")

print("\nQuality Metrics:")
metrics = analyze_chunks(sample_chunks)
print(f"  Total Chunks: {metrics['total_chunks']}")
print(f"  Average Length: {metrics['avg_length']:.0f} characters")
print(f"  Std Dev Length: {metrics['std_length']:.0f} characters")
print(f"  Min Length: {metrics['min_length']} characters")
print(f"  Max Length: {metrics['max_length']} characters")
print(f"  Average Overlap: {metrics['avg_overlap']:.0f} characters")
print(f"  Information Density: {metrics['avg_density']:.3f}")

print("\nInterpretation:")
if metrics['std_length'] / metrics['avg_length'] > 0.5:
    print("  ⚠️ High variance in chunk lengths")
else:
    print("  ✓ Chunk lengths are uniform")

if metrics['avg_density'] > 0.6:
    print("  ✓ High information density")
else:
    print("  ⚠️ Low information density")

print("\n" + "=" * 60)
print("Chunk quality test completed!")
