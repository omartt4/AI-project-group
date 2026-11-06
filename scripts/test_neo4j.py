import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI", "")
# Aura behind SSL inspection / antivirus: skip strict cert verification
uri = uri.replace("neo4j+s://", "neo4j+ssc://")

user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print("Connecting to:", uri)
print("Username:", user)

driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    driver.verify_connectivity()
    with driver.session() as session:
        for r in session.run("MATCH (n) RETURN labels(n) AS type, count(*) AS total"):
            print(r["type"], r["total"])
    print("✓ Connected")
except Exception as e:
    print("✗ Connection failed:", e)
finally:
    driver.close()