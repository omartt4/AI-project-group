#!/usr/bin/env python3
import chromadb
from lab_common import get_config

config = get_config()
client = chromadb.PersistentClient(path=config.chroma_dir)
print("Chroma path:", config.chroma_dir)
print("Collections:")
for c in client.list_collections():
    print("  " + c.name + ":", c.count())
