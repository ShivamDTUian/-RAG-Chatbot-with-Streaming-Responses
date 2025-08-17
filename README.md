# 📙RAG Document Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) Chatbot** that can answer questions based on a set of uploaded documents.  
It uses **Sentence Transformers** for embeddings, **FAISS** for vector search, and **Flan-T5** for language generation.  
The chatbot is deployed with a **Streamlit UI** where users can ask queries, view streaming responses, and export chat history.

---

## Why RAG?

Large Language Models (LLMs) like GPT or Flan-T5 are powerful but often **hallucinate** when asked about external knowledge.  
RAG solves this problem by **retrieving context from documents** and grounding the LLM’s response in real data.  

**In short:**  
. Embeddings + Vector DB → find relevant document chunks  
. LLM → generates an answer using retrieved context  

---

## Project Architecture & Flow


flowchart TD
    A[Raw Documents] --> B[Document Processor - Chunking]
    B --> C[Embeddings Generator - SentenceTransformer]
    C --> D[Vector Database - FAISS Index]
    D --> E[RAG Pipeline - Flan-T5]
    E --> F[Streamlit Chat UI with Export]


**##🛠️ Installation & Setup**
1. Install Dependencies
    pip install -r requirements.txt


3. Run Document Preprocessing
    python src/document_processor.py


✔️ Splits documents into chunks
✔️ Creates chunks/processed_chunks.json


4. Generate Embeddings
    python src/embeddings_generator.py


✔️ Generates embeddings for chunks
✔️ Saves them in vectordb/


5. Launch Chatbot
    streamlit run app.py


Now open your browser at http://localhost:8501 🎉



**Step-by-Step Flow**


1. Document Preprocessing

    Input: Raw PDF/TXT documents
    
    Output: Small text chunks (~300–500 words each) stored in processed_chunks.json

2. Embeddings Generation

    Model: all-MiniLM-L6-v2
    
    Converts each chunk into a 384-dimensional vector
    
    Saves results in vectordb/embeddings.npy and vectordb/metadata.json
   

4. Vector Database (FAISS)

    Creates FAISS index for fast similarity search
    
    Retrieves top-k most relevant chunks for each user query
   

6. RAG Pipeline (Flan-T5)

    Takes query + retrieved chunks → creates prompt
    
    Uses google/flan-t5-base to generate grounded answer
   

8. Streamlit Chatbot

    Interactive web interface
    
    Streams answers in real-time
    
    Displays sources of answer
    
    Allows chat history export (.json)
   


# 🤖 Model & Embedding Choices

Embedding Model:
  **all-MiniLM-L6-v2**
   . Produces 384-d embeddings
   . Optimized for semantic similarity
   . Fast + lightweight (ideal for RAG pipelines)
   

Language Model:
  **google/flan-t5-base**
   . Instruction-tuned model
   . Great for concise Q&A
   . ~512 token input, ~250 token output
   

Vector DB:
  **FAISS**
   . Efficient similarity search
   . L2-normalized inner product search


**Demo**


**Video**: Demo Link : (https://drive.google.com/file/d/1THu2l97pRTZIfm4zSqO9tnfJhkBDV0fk/view?usp=sharing)



**GitHub Repo**: rag-chatbot:  (https://github.com/ShivamDTUian/-RAG-Chatbot-with-Streaming-Responses)
