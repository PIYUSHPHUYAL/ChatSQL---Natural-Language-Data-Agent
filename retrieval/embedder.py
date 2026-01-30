"""
Embedder - Generates semantic embeddings for text
Uses Sentence Transformers (local, free)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os


class SchemaEmbedder:
    """
    Generates embeddings for database schemas.
    Uses all-MiniLM-L6-v2 model (lightweight, fast, good quality)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with specific model.

        Args:
            model_name: HuggingFace model name
                       all-MiniLM-L6-v2: Fast, 384 dimensions, good quality
        """
        self.model_name = model_name
        print(f"🔄 Loading embedding model: {model_name}...")

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"✅ Model loaded! Embedding dimension: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            NumPy array of shape (dimension,)
        """
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (more efficient).

        Args:
            texts: List of texts to embed

        Returns:
            NumPy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1, vec2: NumPy arrays of same shape

        Returns:
            Similarity score between -1 and 1 (1 = identical)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Dot product of normalized vectors = cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to disk"""
        np.save(filepath, embeddings)
        print(f"💾 Embeddings saved to: {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from disk"""
        embeddings = np.load(filepath)
        print(f"📂 Embeddings loaded from: {filepath}")
        return embeddings


def test_embedder():
    """Test embedding generation and similarity"""
    print("Testing Schema Embedder...\n")

    # Initialize
    embedder = SchemaEmbedder()

    # Test texts (similar and dissimilar)
    text1 = "Table stores whale trade alerts with prices and values"
    text2 = "Database table containing large cryptocurrency transactions"
    text3 = "The weather is sunny today"

    print("\nGenerating embeddings...")
    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)
    emb3 = embedder.embed_text(text3)

    print(f"Embedding shape: {emb1.shape}")
    print(f"Embedding dimension: {embedder.dimension}")

    # Calculate similarities
    print("\nCalculating similarities:")
    sim_12 = embedder.cosine_similarity(emb1, emb2)
    sim_13 = embedder.cosine_similarity(emb1, emb3)
    sim_23 = embedder.cosine_similarity(emb2, emb3)

    print(f"  Text 1 vs Text 2 (similar topics): {sim_12:.3f}")
    print(f"  Text 1 vs Text 3 (different topics): {sim_13:.3f}")
    print(f"  Text 2 vs Text 3 (different topics): {sim_23:.3f}")

    # Test batch embedding
    print("\nTesting batch embedding...")
    batch_texts = [text1, text2, text3]
    batch_embeddings = embedder.embed_batch(batch_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")

    # Test save/load
    print("\nTesting save/load...")
    embedder.save_embeddings(batch_embeddings, 'test_embeddings.npy')
    loaded = embedder.load_embeddings('test_embeddings.npy')
    print(f"Loaded shape: {loaded.shape}")

    # Cleanup
    os.remove('test_embeddings.npy')

    print("\n✅ Embedder test complete!")
    print("\n💡 Key insight: Similar texts have similarity ~0.6-0.8")
    print("   Different topics have similarity ~0.0-0.3")


if __name__ == "__main__":
    test_embedder()