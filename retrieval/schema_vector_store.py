"""
Schema Vector Store - Loads schema embeddings into custom vector store
Provides semantic search interface for database schemas
"""

import numpy as np
import json
from pathlib import Path
import sys

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.vector_store import VectorStore
from retrieval.embedder import SchemaEmbedder


class SchemaVectorStore:
    """
    Wrapper around VectorStore specifically for database schemas.
    Provides convenient methods for schema search.
    """

    def __init__(self):
        self.vector_store = None
        self.embedder = None

    def load_from_files(
        self,
        embeddings_file: str = 'config/schema_embeddings.npy',
        metadata_file: str = 'config/schema_metadata.json'
    ):
        """
        Load pre-computed schema embeddings into vector store.

        Args:
            embeddings_file: Path to saved embeddings (.npy)
            metadata_file: Path to saved metadata (.json)
        """
        print("\n" + "="*60)
        print("Loading Schema Vector Store")
        print("="*60 + "\n")

        # Load embeddings
        print(f"📂 Loading embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        print(f"  Shape: {embeddings.shape}")

        # Load metadata
        print(f"📂 Loading metadata from {metadata_file}...")
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
        print(f"  Tables: {len(metadata_list)}")

        # Create vector store
        dimension = embeddings.shape[1]
        self.vector_store = VectorStore(dimension=dimension)

        # Add all vectors
        print(f"\n➕ Adding {len(metadata_list)} tables to vector store...")
        self.vector_store.add_batch(embeddings, metadata_list)

        print(f"\n✅ Schema vector store ready!")
        print(f"   - {len(self.vector_store)} tables indexed")
        print(f"   - Dimension: {dimension}")

    def load_embedder(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load embedder for query encoding"""
        print(f"\n🔄 Loading embedder: {model_name}")
        self.embedder = SchemaEmbedder(model_name=model_name)

    def search_schemas(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.2
    ) -> list:
        """
        Search for relevant database schemas using natural language.

        Args:
            query: Natural language question
            top_k: Number of tables to return
            min_score: Minimum similarity threshold

        Returns:
            List of relevant tables with scores
        """
        if self.embedder is None:
            raise RuntimeError("Embedder not loaded. Call load_embedder() first.")

        # Encode query
        query_embedding = self.embedder.embed_text(query)

        # Search
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            min_score=min_score
        )

        return results

    def print_search_results(self, query: str, results: list):
        """Pretty print search results"""
        print(f"\n🔍 Query: '{query}'")
        print(f"   Found {len(results)} relevant table(s):\n")

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            score = result['score']

            print(f"   {i}. {metadata['table_name']:20} (similarity: {score:.3f})")

            # Show columns
            columns = metadata['columns']
            col_names = [c['name'] for c in columns[:5]]  # First 5 columns
            if len(columns) > 5:
                col_names.append(f"... +{len(columns)-5} more")
            print(f"      Columns: {', '.join(col_names)}")
            print()


def demo_schema_search():
    """Demonstrate schema vector store with real queries"""

    print("\n" + "="*60)
    print("SCHEMA VECTOR STORE DEMO")
    print("="*60)

    # Initialize
    store = SchemaVectorStore()

    # Load schema embeddings
    store.load_from_files()

    # Load embedder
    store.load_embedder()

    print("\n" + "="*60)
    print("Testing Natural Language Schema Search")
    print("="*60)

    # Test queries
    test_queries = [
        "show me all whale transactions",
        "what tables have price information?",
        "find trading volume data",
        "where is whale detection configuration stored?",
        "get latest crypto prices",
        "show me trade statistics",
    ]

    for query in test_queries:
        results = store.search_schemas(query, top_k=2)
        store.print_search_results(query, results)

    print("\n" + "="*60)
    print("Interactive Search Mode")
    print("="*60)
    print("\nTry your own queries! (type 'quit' to exit)\n")

    while True:
        try:
            user_query = input("🔍 Your question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_query:
                continue

            results = store.search_schemas(user_query, top_k=3)
            store.print_search_results(user_query, results)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    demo_schema_search()