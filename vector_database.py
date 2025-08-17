"""
Vector Database Manager for RAG Chatbot
----------------------------------------------------

This file handles:
1. Loading embeddings and metadata
2. Creating FAISS index for fast similarity search
3. Searching for relevant chunks based on user queries
4. Managing the vector database operations
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    A class to manage the vector database for document retrieval
    """
    def __init__(self, vectordb_folder="vectordb", model_name="all-MiniLM-L6-v2"):
        self.vectordb_folder = vectordb_folder
        self.model_name = model_name
        
        self.index = None
        self.chunks = None
        self.embeddings = None
        self.embedding_model = None
        
        self._load_database()
    
    def _load_database(self):
        try:
            embeddings_file = os.path.join(self.vectordb_folder, 'embeddings.npy')
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            
            metadata_file = os.path.join(self.vectordb_folder, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks")
            
            model_info_file = os.path.join(self.vectordb_folder, 'model_info.json')
            if os.path.exists(model_info_file):
                with open(model_info_file, 'r') as f:
                    model_info = json.load(f)
                    if model_info['model_name'] != self.model_name:
                        logger.warning(f"Model mismatch! Expected {self.model_name}, got {model_info['model_name']}")
            
            if self.embeddings is not None:
                self._create_faiss_index()
            
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("Vector database loaded successfully...!")
        except Exception as e:
            logger.error(f"❌ Error loading database: {e}")
            logger.info("Please run embeddings_generator.py first!")
    

    def _create_faiss_index(self):
        if self.embeddings is None:
            logger.error("❌ No embeddings available!")
            return
        
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        logger.info(f"🗃️  Created FAISS index with {self.index.ntotal} vectors")
        index_file = os.path.join(self.vectordb_folder, 'faiss_index.bin')
        faiss.write_index(self.index, index_file)
        logger.info(f"Saved FAISS index to {index_file}")

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Searching for relevant chunks based on a query
        """
        if self.index is None or self.embedding_model is None:
            logger.error("❌ Database not properly loaded!")
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        # DEBUG PRINT
        print("\n[DEBUG] FAISS Scores and Indices for query:", query)
        for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            idx = int(idx.item())  #Fixing numpy.int64 to Python int
            preview = self.chunks[idx]['content'][:120]
            print(f"{i}. Score: {score:.3f} | Preview: {preview}")
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            idx = int(idx.item()) 
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = float(score)
            chunk['rank'] = i + 1
            results.append(chunk)

        logger.info(f"🔍 Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results

    def get_context_for_query(self, query: str, max_context_length: int = 5000, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Get relevant context for a query, formatted for prompt injection
        """
        relevant_chunks = self.search(query, top_k=top_k)
        if not relevant_chunks:
            return "No relevant information found.", []
        
        print("\n[DEBUG] Chunks being used for prompt context for query:", query)
        for chunk in relevant_chunks:
            print(f"Score: {chunk['similarity_score']:.3f}, Preview: {chunk['content'][:100]}")
        
        context_parts = []
        current_length = 0
        used_chunks = []
        for chunk in relevant_chunks:
            chunk_text = f"Source: {chunk['source_file']}\nContent: {chunk['content']}\n---\n"
            if current_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            used_chunks.append(chunk)
        context = "\n".join(context_parts)
        logger.info(f"Built context with {len(used_chunks)} chunks ({current_length} chars)")
        return context, used_chunks
    
    def get_database_stats(self) -> Dict:
        stats = {
            'total_chunks': len(self.chunks) if self.chunks else 0,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_size': self.index.ntotal if self.index else 0,
            'model_name': self.model_name,
            'database_loaded': self.index is not None
        }
        if self.chunks:
            word_counts = [chunk['word_count'] for chunk in self.chunks]
            stats.update({
                'avg_chunk_size': np.mean(word_counts),
                'min_chunk_size': np.min(word_counts),
                'max_chunk_size': np.max(word_counts),
                'total_documents': len(set(chunk['source_file'] for chunk in self.chunks))
            })
        return stats

    def rebuild_index(self):
        if self.embeddings is not None:
            logger.info("Rebuilding FAISS index...")
            self._create_faiss_index()
        else:
            logger.error("❌ No embeddings available to rebuild index!")

def main():
    vdb = VectorDatabase()
    stats = vdb.get_database_stats()

    print("Vector Database Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total documents: {stats.get('total_documents', 'N/A')}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Average chunk size: {stats.get('avg_chunk_size', 'N/A'):.1f} words")
    print(f"   Model: {stats['model_name']}")
    print(f"   Database loaded: {stats['database_loaded']}")

    if not stats['database_loaded']:
        print("\n❌ Database not loaded!")
        print("Please run document_processor.py and embeddings_generator.py first!")
        return

    test_queries = [
        "What are the payment policies?",
        "How do I cancel an order?",
        "What are the dispute resolution procedures?",
        "Tell me about user agreements"
    ]

    print("\nTesting with sample queries:")
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vdb.search(query, top_k=3)
        if results:
            print(f"   Found {len(results)} relevant chunks:")
            for i, chunk in enumerate(results, 1):
                print(f"   {i}. Score: {chunk['similarity_score']:.3f} | Source: {chunk['source_file']}")
                print(f"      Preview: {chunk['content'][:100]}...")
        else:
            print("   No relevant chunks found.")
        context, sources = vdb.get_context_for_query(query, max_context_length=800)
        print(f"   Context length: {len(context)} characters from {len(sources)} sources")

    print("\n✅ Vector database test complete!")

if __name__ == "__main__":
    main()
