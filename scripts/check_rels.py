"""Quick script to check relationships in the graph."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo4j import GraphDatabase
from graph_rag.config import settings

driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    connection_timeout=30.0,
)

with driver.session() as session:
    # Check all relationship types between entities
    result = session.run("""
        MATCH (s:Entity {dataset_id: $dataset_id})-[r]->(t:Entity {dataset_id: $dataset_id})
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
    """, dataset_id=settings.DATASET_ID)
    
    print('Entity-to-Entity relationships:')
    for record in result:
        print(f'  {record["rel_type"]}: {record["count"]}')
    
    # Check if there are any relationships at all
    result = session.run("""
        MATCH (s:Entity {dataset_id: $dataset_id})-[r]->(t)
        WHERE t.dataset_id = $dataset_id
        RETURN type(r) as rel_type, labels(t) as target_labels, count(r) as count
        ORDER BY count DESC
        LIMIT 20
    """, dataset_id=settings.DATASET_ID)
    
    print('\nAll relationships from Entity nodes:')
    for record in result:
        print(f'  {record["rel_type"]} -> {record["target_labels"]}: {record["count"]}')

driver.close()
