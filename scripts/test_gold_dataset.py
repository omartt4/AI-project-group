"""
Test Gold Dataset Structure.

Tests the gold test dataset creation and statistics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

class QueryLanguage(Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    MIXED = "mixed"

class QueryType(Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    RELATIONAL = "relational"
    EXPLORATORY = "exploratory"

@dataclass
class GoldTestQuery:
    query_id: str
    query: str
    language: QueryLanguage
    query_type: QueryType
    relevant_doc_ids: List[str] = field(default_factory=list)
    expected_answer: str = ""

@dataclass
class GoldTestDataset:
    name: str
    description: str
    queries: List[GoldTestQuery] = field(default_factory=list)
    
    def add_query(self, query: GoldTestQuery):
        self.queries.append(query)
    
    def statistics(self) -> Dict[str, Any]:
        return {
            "total_queries": len(self.queries),
            "english": len([q for q in self.queries if q.language == QueryLanguage.ENGLISH]),
            "arabic": len([q for q in self.queries if q.language == QueryLanguage.ARABIC]),
            "mixed": len([q for q in self.queries if q.language == QueryLanguage.MIXED]),
        }

# Create sample dataset
dataset = GoldTestDataset(
    name="Test Dataset",
    description="Sample gold test dataset",
)

dataset.add_query(GoldTestQuery(
    query_id="en_001",
    query="What is attention?",
    language=QueryLanguage.ENGLISH,
    query_type=QueryType.FACTUAL,
    expected_answer="Attention is a mechanism...",
))

dataset.add_query(GoldTestQuery(
    query_id="ar_001",
    query="ما هو الانتباه؟",
    language=QueryLanguage.ARABIC,
    query_type=QueryType.FACTUAL,
    expected_answer="الانتباه هو آلية...",
))

dataset.add_query(GoldTestQuery(
    query_id="mixed_001",
    query="What is الانتباه؟",
    language=QueryLanguage.MIXED,
    query_type=QueryType.FACTUAL,
    expected_answer="Attention is...",
))

print("Gold Dataset Test")
print("=" * 60)

print(f"\nDataset: {dataset.name}")
print(f"Description: {dataset.description}")

print("\nQueries:")
for q in dataset.queries:
    print(f"  {q.query_id}: {q.query} ({q.language.value})")

print("\nStatistics:")
stats = dataset.statistics()
print(f"  Total Queries: {stats['total_queries']}")
print(f"  English: {stats['english']}")
print(f"  Arabic: {stats['arabic']}")
print(f"  Mixed: {stats['mixed']}")

print("\n" + "=" * 60)
print("Gold dataset test completed!")
