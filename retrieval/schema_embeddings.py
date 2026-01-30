"""
Schema Embeddings - Generates and saves embeddings for database schemas
Combines schema_loader and embedder to create searchable schema vectors
"""

import json
import numpy as np
import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.schema_loader import SchemaLoader
from retrieval.embedder import SchemaEmbedder
from config.database import DatabaseConfig


class SchemaEmbeddingGenerator:
    """
    Generates embeddings for database schemas.
    Creates searchable vectors for semantic retrieval.
    """

    def __init__(self):
        self.embedder = SchemaEmbedder()
        self.schema_data = None
        self.embeddings = None
        self.metadata = []

    def load_schema(self, schema_file: str = 'config/schema.json'):
        """Load schema from JSON file"""
        with open(schema_file, 'r') as f:
            self.schema_data = json.load(f)

        print(f"📂 Loaded schema for {len(self.schema_data['tables'])} tables")

    def extract_and_embed_schemas(self):
        """
        Extract schema and generate embeddings.
        Connects to DB, extracts schema, creates embeddings.
        """
        print("\n" + "="*60)
        print("STEP 1: Extract Database Schema")
        print("="*60 + "\n")

        # Create schema loader
        loader = SchemaLoader(
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            database=DatabaseConfig.DATABASE,
            user=DatabaseConfig.USER,
            password=DatabaseConfig.PASSWORD
        )

        loader.connect()
        schema_data = loader.extract_full_schema()
        loader.save_schema(schema_data, 'config/schema.json')
        loader.close()

        self.schema_data = schema_data

        print("\n" + "="*60)
        print("STEP 2: Generate Embeddings for Each Table")
        print("="*60 + "\n")

        # Prepare texts to embed
        texts_to_embed = []

        for table_name, table_info in self.schema_data['tables'].items():
            # Use the full description (includes columns + purpose)
            text = table_info['description']

            texts_to_embed.append(text)

            # Store metadata for later retrieval
            self.metadata.append({
                'table_name': table_name,
                'columns': table_info['columns'],
                'description_text': text
            })

            print(f"  Prepared: {table_name}")

        # Generate embeddings (batch is more efficient)
        print(f"\n🔄 Generating embeddings for {len(texts_to_embed)} tables...")
        self.embeddings = self.embedder.embed_batch(texts_to_embed)

        print(f"✅ Generated embeddings: shape {self.embeddings.shape}")

    def save_embeddings(self, output_dir: str = 'config'):
        """Save embeddings and metadata to disk"""
        print(f"\n💾 Saving embeddings to {output_dir}/...")

        # Save embeddings (NumPy array)
        embeddings_file = f"{output_dir}/schema_embeddings.npy"
        np.save(embeddings_file, self.embeddings)
        print(f"  ✅ Embeddings: {embeddings_file}")

        # Save metadata (JSON)
        metadata_file = f"{output_dir}/schema_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"  ✅ Metadata: {metadata_file}")

    def test_search(self, query: str):
        """
        Test semantic search on embeddings.

        Args:
            query: Natural language question
        """
        print(f"\n🔍 Testing search: '{query}'")

        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Calculate similarities with all table embeddings
        similarities = []
        for i, table_embedding in enumerate(self.embeddings):
            sim = self.embedder.cosine_similarity(query_embedding, table_embedding)
            similarities.append((i, sim))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Show top 3 results
        print(f"\nTop 3 relevant tables:")
        for rank, (idx, score) in enumerate(similarities[:3], 1):
            table_name = self.metadata[idx]['table_name']
            print(f"  {rank}. {table_name:20} (similarity: {score:.3f})")


def main():
    """Main workflow: Extract → Embed → Save → Test"""

    print("\n" + "="*60)
    print("SCHEMA EMBEDDING GENERATOR")
    print("="*60)

    generator = SchemaEmbeddingGenerator()

    # Extract schema and generate embeddings
    generator.extract_and_embed_schemas()

    # Save to disk
    generator.save_embeddings()

    # Test searches
    print("\n" + "="*60)
    print("STEP 3: Test Semantic Search")
    print("="*60)

    test_queries = [
        "show me whale trades",
        "what are the current cryptocurrency prices?",
        "trading statistics and volumes",
        "configuration for whale detection"
    ]

    for query in test_queries:
        generator.test_search(query)

    print("\n" + "="*60)
    print("✅ COMPLETE! Schema embeddings ready for agent use")
    print("="*60)

    print("\n📁 Files created:")
    print("  - config/schema.json (table structures)")
    print("  - config/schema_embeddings.npy (vector embeddings)")
    print("  - config/schema_metadata.json (table metadata)")

    print("\n💡 Next: Build vector store for fast semantic search!")


if __name__ == "__main__":
    main()