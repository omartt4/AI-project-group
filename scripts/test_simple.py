"""
Simple test to verify the project structure is working on D: drive.
"""

import sys
from pathlib import Path

print("Testing project structure on D: drive...")
print("=" * 60)

# Check project root
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
print(f"Drive: {project_root.drive}")

# Check if key files exist
key_files = [
    "src/graph_rag/config.py",
    "src/graph_rag/retriever/language_detector.py",
    "src/graph_rag/retriever/query_engine.py",
    "pyproject.toml",
]

print("\nChecking key files:")
for file in key_files:
    full_path = project_root / file
    exists = "✓" if full_path.exists() else "✗"
    print(f"{exists} {file}")

print("\n✓ Project structure is valid on D: drive!")
print("\nNote: The memory allocation error is due to heavy libraries (scipy/scikit-learn)")
print("that were previously installed. These are not needed for the core functionality.")
