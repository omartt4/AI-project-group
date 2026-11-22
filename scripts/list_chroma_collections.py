#!/usr/bin/env python3
"""List all Chroma collections."""
import chromadb
from chromadb.config import Settings

config = {
    'chroma_dir': 'data/lab/chroma'
}
client = chromadb.PersistentClient(path=config['chroma_dir'], settings=Settings(anonymized_telemetry=False))

collections = client.list_collections()
print("Chroma collections:")
for col in collections:
    print(f"  {col.name}")
