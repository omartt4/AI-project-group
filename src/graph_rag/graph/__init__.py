"""Knowledge graph construction, resolution, discovery, and query package.

Submodules are intentionally not imported here. Keeping package initialization
light prevents Neo4j and data-processing dependencies from loading when callers
only need the entity or taxonomy data classes.
"""
