# Multilingual Embedding Model Setup Guide

This guide explains how to configure your Graph RAG system for Arabic and English support using the BGE-M3 multilingual embedding model.

## Overview

**BGE-M3** (BAAI/bge-m3) is a state-of-the-art multilingual embedding model that:
- Supports 100+ languages including Arabic and English
- Optimized for cross-lingual retrieval
- Excellent performance on multilingual RAG tasks
- Available for free via Ollama

## Quick Setup

### Option 1: Using the Setup Script (Recommended)

```bash
uv run python scripts/setup_multilingual_embeddings.py
```

This script will:
- Check if Ollama is running
- Verify if BGE-M3 is already installed
- Pull the model if needed (~2.2GB download)

### Option 2: Manual Setup

```bash
# Pull the model directly from Ollama
ollama pull bge-m3
```

## Configuration

### Step 1: Update .env File

Edit your `.env` file and set:

```env
OLLAMA_EMBED_MODEL=bge-m3
```

### Step 2: Re-index Your Documents

After changing the embedding model, you must re-index your documents:

```bash
uv run python scripts/ingest.py
```

**Important:** The old ChromaDB embeddings are incompatible with the new model. The ingestion script will rebuild the vector index with BGE-M3 embeddings.

## Model Comparison

| Model | Languages | Size | Use Case |
|-------|-----------|------|----------|
| **bge-m3** | 100+ (Arabic, English, etc.) | ~2.2GB | **Multilingual RAG** (recommended) |
| nomic-embed-text | English only | ~274MB | English-only RAG |

## Testing Multilingual Support

After setup, test with Arabic queries:

```python
# Example: Query in Arabic
query_arabic = "ما هو التعلم العميق؟"  # What is deep learning?
```

The system will:
1. Embed the Arabic query using BGE-M3
2. Retrieve relevant English documents
3. Generate an answer in the appropriate language

## Troubleshooting

### Ollama Server Not Running

```bash
# Start Ollama
ollama serve
```

### Model Pull Fails

Ensure you have sufficient disk space (~3GB for BGE-M3) and stable internet connection.

### Re-indexing Takes Too Long

BGE-M3 embeddings are larger than nomic-embed-text. For large document sets, re-indexing may take longer. This is normal and one-time cost.

### Fallback to English-Only

If you encounter issues, you can revert to English-only by setting in `.env`:

```env
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Then re-index your documents again.

## Performance Notes

- **Embedding Dimension:** BGE-M3 produces 1024-dimensional vectors (vs 768 for nomic-embed-text)
- **Memory Usage:** Higher memory footprint during embedding generation
- **Retrieval Quality:** Significantly better for cross-lingual queries
- **Latency:** Slightly higher embedding generation time (~10-20% slower)

## Next Steps

After completing multilingual setup:
1. Test with Arabic queries
2. Evaluate cross-language retrieval quality
3. Prepare gold test dataset with bilingual queries (see Task 8)
