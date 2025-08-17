"""
RAG Pipeline
------------------------------------------------------------
This script ties together:
1. The Vector Database (retrieval)
2. A language model (Flan-T5 for generation)

Flow:
- Fetch top-k relevant chunks from the vector DB
- Build a context-aware prompt
- Generate a concise answer using the LLM
"""


import logging
from typing import Dict, Any
from transformers import pipeline
from vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        
        try:
            logger.info("Initializing Vector Database...")
            self.vector_db = VectorDatabase()  # auto-loads embeddings & index
            logger.info("Vector Database initialized")

            logger.info("Loading Language Model (google/flan-t5-base)...")
            self.generator = pipeline(
                task="text2text-generation",
                model="google/flan-t5-base"
            )
            logger.info("Language Model loaded successfully")

            self.conversation_turns = 0

        except Exception as e:
            logger.error(f"❌ Error initializing RAGPipeline: {e}")
            raise

    def answer_question(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        #Retrieve context from vector DB and generate an answer.
        try:
            self.conversation_turns += 1

            # Retrieve relevant context
            context, sources = self.vector_db.get_context_for_query(query, top_k=3)

            if len(context) > 3500:
                context = context[:3500]

            # Prompt
            prompt = (
                f"You are a helpful assistant. Use only the context below to answer the question. "
                f"Give a **concise answer (2-3 sentences max)** in simple language. "
                f"If the context does not have the answer, say 'I could not find this information in the documents.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            # 3) Generate
            outputs = self.generator(
                prompt,
                max_length=250,
                num_return_sequences=1,
                do_sample=False,
            )
            answer = outputs[0]["generated_text"].strip()
            if not answer:
                answer = "I could not find this information in the documents."

            return {
                "answer": answer,
                "sources": sources if include_sources else [],
                "confidence": 0.75,  # simple fixed confidence for now
            }

        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return {
                "answer": "I encountered an error while trying to answer.",
                "sources": [],
                "confidence": 0.0,
            }

    def clear_conversation_history(self):
        """Reset conversation count."""
        self.conversation_turns = 0

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return system stats for sidebar display."""
        return {
            "conversation_turns": self.conversation_turns,
            "model_name": "google/flan-t5-base",
            "database_stats": self.vector_db.get_database_stats(),
        }
