"""
Retrieval module - Public API for schema search
Provides easy-to-use functions for agent integration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.schema_vector_store import SchemaVectorStore

# Global instance (lazy loaded)
_schema_store = None


def get_schema_store() -> SchemaVectorStore:
    """
    Get or create the global schema vector store instance.
    Singleton pattern for efficiency.
    """
    global _schema_store

    if _schema_store is None:
        _schema_store = SchemaVectorStore()
        _schema_store.load_from_files()
        _schema_store.load_embedder()

    return _schema_store


def search_relevant_tables(query: str, top_k: int = 3) -> list:
    """
    Simple API: Search for relevant tables given a natural language query.

    Args:
        query: Natural language question
        top_k: Number of tables to return

    Returns:
        List of dicts with table info and similarity scores

    Example:
        >>> results = search_relevant_tables("show me whale trades")
        >>> print(results[0]['metadata']['table_name'])
        'alerts'
    """
    store = get_schema_store()
    return store.search_schemas(query, top_k=top_k)


def get_table_info(table_name: str) -> dict:
    """
    Get detailed information about a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Dict with columns and description
    """
    store = get_schema_store()

    # Search through metadata
    for metadata in store.vector_store.metadata:
        if metadata['table_name'] == table_name:
            return metadata

    return None


# Quick test
if __name__ == "__main__":
    print("Testing retrieval API...\n")

    # Test 1: Search
    print("Test 1: Search for whale trades")
    results = search_relevant_tables("show me whale trades", top_k=2)
    for r in results:
        print(f"  - {r['metadata']['table_name']} (score: {r['score']:.3f})")

    # Test 2: Get table info
    print("\nTest 2: Get info for 'alerts' table")
    info = get_table_info('alerts')
    if info:
        print(f"  Table: {info['table_name']}")
        print(f"  Columns: {len(info['columns'])}")
        print(f"  First 3: {[c['name'] for c in info['columns'][:3]]}")

    print("\n✅ Retrieval API ready for agent integration!")