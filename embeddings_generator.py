"""
Embeddings Builder for RAG Chatbot
-----------------------------------

This script:
1. Reads processed text chunks
2. Converts them into embeddings using a pre-trained transformer
3. Stores embeddings + metadata for later vector search

Why embeddings?
- Each text gets turned into a vector (list of numbers)
- Similar texts → similar vectors
- Useful for semantic search in RAG pipeline
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# --- Logger ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Generate and manage embeddings for document chunks."""

    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chunks_dir: str = "chunks",
                 db_dir: str = "vectordb"):
        self.model_name = model_name
        self.chunks_dir = chunks_dir
        self.db_dir = db_dir

        os.makedirs(self.db_dir, exist_ok=True)

        log.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        log.info("✅ Model ready.")

    # -------- Loading --------
    def load_chunks(self) -> List[Dict]:
        """Read chunks JSON file created by document_processor.py."""
        path = os.path.join(self.chunks_dir, "processed_chunks.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}. Run document_processor.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        log.info(f"📖 Loaded {len(chunks)} chunks")
        return chunks

    # -------- Embedding --------
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Convert list of texts into embeddings using mini-batching."""
        log.info(f"🔄 Encoding {len(texts)} texts...")
        vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            vecs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            vectors.append(vecs)
        embeddings = np.vstack(vectors)
        log.info(f"✅ Created embeddings with shape {embeddings.shape}")
        return embeddings

    # -------- Saving --------
    def save_outputs(self, chunks: List[Dict], embeddings: np.ndarray):
        """Save embeddings, metadata, and a summary CSV."""
        # Embeddings
        npy_path = os.path.join(self.db_dir, "embeddings.npy")
        np.save(npy_path, embeddings)
        log.info(f"💾 Embeddings saved → {npy_path}")

        # Metadata
        meta_path = os.path.join(self.db_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        log.info(f"💾 Metadata saved → {meta_path}")

        # Model info
        model_info = {
            "model": self.model_name,
            "vector_dim": embeddings.shape[1],
            "chunks_count": len(chunks),
            "max_seq_len": self.model.max_seq_length
        }
        info_path = os.path.join(self.db_dir, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        log.info(f"💾 Model info saved → {info_path}")

        # Summary CSV
        df = pd.DataFrame({
            "chunk_id": [c["chunk_id"] for c in chunks],
            "source": [c["source_file"] for c in chunks],
            "words": [c["word_count"] for c in chunks],
            "preview": [c["content"][:100] + "..." if len(c["content"]) > 100 else c["content"]
                        for c in chunks]
        })
        csv_path = os.path.join(self.db_dir, "chunks_summary.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"💾 Summary CSV saved → {csv_path}")

    # -------- Full Pipeline --------
    def process(self):
        """Run full embedding workflow: load chunks → embed → save."""
        chunks = self.load_chunks()
        if not chunks:
            log.error("❌ No chunks found.")
            return

        texts = [c["content"] for c in chunks]
        embeddings = self.embed_texts(texts)
        self.save_outputs(chunks, embeddings)

        print(f"\n✅ Embeddings complete!")
        print(f"📊 {len(embeddings)} vectors | Dim: {embeddings.shape[1]}")
        print(f"🤖 Model: {self.model_name}")
        print(f"💾 Files in '{self.db_dir}' folder")

    # -------- Quick Similarity Test --------
    def test_query(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        """Check nearest chunks for a sample query."""
        emb_path = os.path.join(self.db_dir, "embeddings.npy")
        meta_path = os.path.join(self.db_dir, "metadata.json")

        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            log.error("❌ No embeddings/metadata found. Run process() first.")
            return []

        embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        q_vec = self.model.encode([query])
        sims = np.dot(embeddings, q_vec.T).flatten()
        top_ids = np.argsort(sims)[-top_k:][::-1]

        results = [(sims[i], chunks[i]) for i in top_ids]
        return results


# --- Run Script ---
def main():
    builder = EmbeddingBuilder()
    print("🔄 Generating embeddings...\n")
    builder.process()

    # Quick demo
    sample = "Explain refund policies"
    print(f"\n🧪 Demo query: {sample}")
    results = builder.test_query(sample, top_k=2)
    for i, (score, chunk) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Source: {chunk['source_file']}")
        print(f"   Text: {chunk['content'][:150]}...")


if __name__ == "__main__":
    main()
