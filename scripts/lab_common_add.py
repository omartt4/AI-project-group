def get_collection_name(model_tag: str, chunk_size: int, chunk_overlap: int, mode: str) -> str:
    """Generate Chroma collection name."""
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{chunk_size}_{chunk_overlap}_{mode}"

