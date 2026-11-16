"""Check paper_id values in relationships."""
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
    # Check paper_id values in relationships
    result = session.run("""
        MATCH (s:Entity {dataset_id: $dataset_id})-[r]->(t:Entity {dataset_id: $dataset_id})
        RETURN DISTINCT r.paper_id as paper_id, count(r) as count
        ORDER BY count DESC
    """, dataset_id=settings.DATASET_ID)
    
    print('Paper IDs in relationships:')
    for record in result:
        print(f'  {record["paper_id"]}: {record["count"]}')
    
    # Check paper UIDs
    result = session.run("""
        MATCH (p:Paper {dataset_id: $dataset_id})
        RETURN p.uid as uid, p.paper_id as paper_id
    """, dataset_id=settings.DATASET_ID)
    
    print('\nPaper nodes:')
    for record in result:
        print(f'  uid: {record["uid"]}')
        print(f'  paper_id: {record["paper_id"]}')

driver.close()
