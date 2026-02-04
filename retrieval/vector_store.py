"""
Custom Vector Store - Built from scratch using NumPy
No external vector databases (Chroma, Pinecone, etc.)
Implements efficient semantic search for schema retrieval
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class VectorStore:
    """
    Custom vector database implementation.
    Stores embeddings with metadata and provides semantic search.

    Features:
    - Fast cosine similarity search
    - Top-K retrieval
    - Metadata storage
    - Persistence (save/load)
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize empty vector store.

        Args:
            dimension: Embedding dimension (must match embedder)
        """
        self.dimension = dimension
        self.vectors = []  # Will become NumPy array
        self.metadata = []  # Parallel list with metadata
        self.num_vectors = 0

        print(f"🗄️  Initialized VectorStore (dimension: {dimension})")

    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add a single vector with metadata.

        Args:
            vector: Embedding vector (shape: dimension,)
            metadata: Associated metadata (table name, description, etc.)

        Returns:
            Index of added vector
        """
        # Validate dimension
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} doesn't match "
                f"store dimension {self.dimension}"
            )

        # Add to storage
        self.vectors.append(vector)
        self.metadata.append(metadata)
        self.num_vectors += 1

        return self.num_vectors - 1

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add multiple vectors at once (more efficient).

        Args:
            vectors: Array of shape (N, dimension)
            metadata_list: List of N metadata dicts

        Returns:
            List of indices for added vectors
        """
        if len(metadata_list) != vectors.shape[0]:
            raise ValueError("Number of vectors must match number of metadata entries")

        indices = []
        for vector, metadata in zip(vectors, metadata_list):
            idx = self.add_vector(vector, metadata)
            indices.append(idx)

        return indices

    def _finalize_vectors(self):
        """Convert vector list to NumPy array for fast operations"""
        if isinstance(self.vectors, list):
            self.vectors = np.array(self.vectors)
            print(f"  Converted to NumPy array: shape {self.vectors.shape}")

    def cosine_similarity(
        self,
        query_vector: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and all vectors.
        Implemented from scratch using NumPy.

        Args:
            query_vector: Shape (dimension,)
            vectors: Shape (N, dimension)

        Returns:
            Similarities: Shape (N,) with values in [-1, 1]
        """
        # Normalize query vector
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        # Normalize all vectors
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        # Dot product = cosine similarity for normalized vectors
        similarities = np.dot(vectors_norm, query_norm)

        return similarities

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar vectors.

        Args:
            query_vector: Query embedding (shape: dimension,)
            top_k: Number of results to return
            min_score: Minimum similarity threshold

        Returns:
            List of dicts with 'metadata', 'score', 'index'
        """
        if self.num_vectors == 0:
            return []

        # Ensure vectors are NumPy array
        self._finalize_vectors()

        # Calculate similarities
        similarities = self.cosine_similarity(query_vector, self.vectors)

        # Get top K indices (argsort returns ascending, so reverse)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            # Filter by minimum score
            if score < min_score:
                continue

            results.append({
                'index': int(idx),
                'score': score,
                'metadata': self.metadata[idx]
            })

        return results

    def save(self, filepath: str):
        """
        Save vector store to disk.

        Args:
            filepath: Path to save (without extension)
        """
        # Ensure vectors are NumPy array
        self._finalize_vectors()

        # Save vectors
        vectors_file = f"{filepath}_vectors.npy"
        np.save(vectors_file, self.vectors)

        # Save metadata
        metadata_file = f"{filepath}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'num_vectors': self.num_vectors,
                'metadata': self.metadata
            }, f, indent=2, default=str)

        print(f"💾 Vector store saved:")
        print(f"  - Vectors: {vectors_file}")
        print(f"  - Metadata: {metadata_file}")

    @classmethod
    def load(cls, filepath: str) -> 'VectorStore':
        """
        Load vector store from disk.

        Args:
            filepath: Path to load from (without extension)

        Returns:
            Loaded VectorStore instance
        """
        # Load metadata first
        metadata_file = f"{filepath}_metadata.json"
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        # Create store
        store = cls(dimension=data['dimension'])

        # Load vectors
        vectors_file = f"{filepath}_vectors.npy"
        store.vectors = np.load(vectors_file)
        store.metadata = data['metadata']
        store.num_vectors = data['num_vectors']

        print(f"📂 Vector store loaded:")
        print(f"  - {store.num_vectors} vectors")
        print(f"  - Dimension: {store.dimension}")

        return store

    def __len__(self) -> int:
        """Return number of vectors in store"""
        return self.num_vectors

    def __repr__(self) -> str:
        return f"VectorStore(dimension={self.dimension}, num_vectors={self.num_vectors})"


def test_vector_store():
    """Test vector store functionality"""
    print("Testing Custom Vector Store...\n")

    # Create store
    dimension = 384  # Same as sentence transformers
    store = VectorStore(dimension=dimension)

    print("\n" + "="*60)
    print("TEST 1: Add Vectors")
    print("="*60)

    # Create some test vectors
    vec1 = np.random.randn(dimension)
    vec2 = np.random.randn(dimension)
    vec3 = vec1 + 0.1 * np.random.randn(dimension)  # Similar to vec1

    # Add with metadata
    store.add_vector(vec1, {'name': 'table_A', 'description': 'First table'})
    store.add_vector(vec2, {'name': 'table_B', 'description': 'Second table'})
    store.add_vector(vec3, {'name': 'table_C', 'description': 'Third table (similar to A)'})

    print(f"Added {len(store)} vectors")
    print(store)

    print("\n" + "="*60)
    print("TEST 2: Search")
    print("="*60)

    # Search using vec1 as query (should find itself + vec3)
    results = store.search(vec1, top_k=3)

    print(f"\nQuery: vec1")
    print(f"Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['metadata']['name']:15} (score: {result['score']:.3f})")

    print("\n" + "="*60)
    print("TEST 3: Save and Load")
    print("="*60)

    # Save
    store.save('test_store')

    # Load
    loaded_store = VectorStore.load('test_store')
    print(f"\nLoaded store: {loaded_store}")

    # Test search on loaded store
    loaded_results = loaded_store.search(vec1, top_k=2)
    print(f"\nSearch on loaded store:")
    for i, result in enumerate(loaded_results, 1):
        print(f"  {i}. {result['metadata']['name']:15} (score: {result['score']:.3f})")

    # Cleanup
    Path('test_store_vectors.npy').unlink()
    Path('test_store_metadata.json').unlink()

    print("\n✅ Vector store test complete!")


if __name__ == "__main__":
    test_vector_store()