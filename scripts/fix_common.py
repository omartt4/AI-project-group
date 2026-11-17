content = open('scripts/lab_common.py', 'r', encoding='utf-8').read()
func = '\ndef get_collection_name(model_tag: str, chunk_size: int, chunk_overlap: int, mode: str) -> str:\n    """Generate Chroma collection name."""\n    safe_tag = model_tag.replace("/", "_").replace(":", "_")\n    return f"retrieval_lab_{safe_tag}_{chunk_size}_{chunk_overlap}_{mode}"\n\n'
idx = content.find('def print_warning')
content = content[:idx] + func + content[idx:]
open('scripts/lab_common.py', 'w', encoding='utf-8').write(content)
print("Fixed")
