"""Streamlit Web App for RAG Chatbot
____________________________________________ """

import numpy as np
import streamlit as st
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our RAG pipeline
try:
    from rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"❌ Error importing RAG pipeline: {e}")
    st.error("Please make sure all required files are in the 'src' folder")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="📚 RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""<style>
[data-testid="stExpander"] details summary {font-weight:bold;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline. Only loads once per session."""
    try:
        with st.spinner(" Loading RAG Pipeline... This may take a few minutes..."):
            rag = RAGPipeline()
        return rag, None
    except Exception as e:
        return None, str(e)

def display_message(role: str, content: str, sources: List[Dict] = None, confidence: float = None):
    """Display a chat message with formatting (for user and assistant)."""
    if role == "user":
        st.markdown(f"""
<div style="background-color:#e8f4fd;padding:8px 14px;border-radius:7px;margin-top:8px;"><b>You:</b> {content}</div>
""", unsafe_allow_html=True)
    else:  # assistant
        conf_class = "confidence-low"
        if confidence is not None:
            if confidence > 0.7:
                conf_class = "confidence-high"
            elif confidence > 0.4:
                conf_class = "confidence-medium"
        confidence_text = f" <span style='color:gray;font-size:90%'>(Confidence: {confidence:.1%})</span>" if confidence is not None else ""
        st.markdown(f"""
<div style="background-color:#f3f9eb;padding:8px 14px;border-radius:7px;margin-top:8px;"><b>Assistant:</b> {content}{confidence_text}</div>
""", unsafe_allow_html=True)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for i, source in enumerate(sources[:5], 1):  # show up to 5
                    st.markdown(
                        f"<div style='border-bottom:1px solid #ececec;padding-bottom:4px;'><b>{i}.</b> <b>Source:</b> {source.get('source_file', 'Unknown')}<br>" +
                        f"<b>Score:</b> {source.get('similarity_score', 0):.3f}<br>" +
                        f"<b>Preview:</b> {source.get('content','')[:180]}{'...' if len(source.get('content',''))>180 else ''}</div>",
                        unsafe_allow_html=True
                    )

st.markdown(
    "<h2 style='font-weight:700;margin-top:0;'>📙 RAG Document Chatbot</h2>"
    "<div style='margin-bottom:18px;font-size:110%'>Ask questions about your documents and get accurate, source-backed answers!</div>",
    unsafe_allow_html=True
)

#### Load RAG pipeline
rag_pipeline, error = load_rag_pipeline()
if error:
    st.error(f"❌ Failed to load RAG pipeline: {error}")
    st.info("""**Troubleshooting Steps:**
1. Make sure you've run `python src/document_processor.py`
2. Then run `python src/embeddings_generator.py`
3. Check that all required packages are installed
4. Ensure your documents are in the `data/` folder
""")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = rag_pipeline

# Sidebar with information and controls
with st.sidebar:
    st.header("System Information")
    stats = rag_pipeline.get_pipeline_stats()
    st.metric("Total Documents", stats['database_stats'].get('total_documents', 'N/A'))
    st.metric("Total Chunks", stats['database_stats']['total_chunks'])
    st.metric("💬 Conversation Turns", stats['conversation_turns'])
    st.subheader("Model Information")
    st.write(f"**Language Model:** {stats['model_name']}")
    st.write(f"**Embedding Model:** {stats['database_stats']['model_name']}")
    st.write(f"**Avg Chunk Size:** {stats['database_stats'].get('avg_chunk_size', 'N/A'):.0f} words")
    st.subheader("Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        rag_pipeline.clear_conversation_history()
        st.success("Chat history cleared!")
        st.rerun()
    if st.button("📊 Show Database Stats"):
        st.json(stats['database_stats'])
    st.subheader("⚙️ Settings")
    st.session_state['show_sources'] = st.checkbox("Show source documents", value=True)
    st.session_state['enable_streaming'] = st.checkbox("Enable streaming responses", value=True)
    st.session_state['max_sources'] = st.slider("Max sources to show", 1, 10, 3)

# Main Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources", []) if st.session_state.get('show_sources', True) else None,
            message.get("confidence"),
        )

with col2:
    st.subheader("💡 Suggested Questions")
    sample_questions = [
        "Main payment policies?",
        "How do I resolve disputes?",
        "What are the user agreement terms?",
        "How can I cancel an order?",
        "Seller requirements?",
        "About refund policies"
    ]
    for question in sample_questions:
        if st.button(question, key=f"sample_{question[:20]}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.experimental_rerun()

    st.subheader("💡 Tips")
    st.info("""
**For best results:**
- Ask specific questions related to documents.  
- Reference document topics (e.g. 'refund', 'payment', 'dispute')  
""")

    # Export Chat
    if st.session_state.messages:
        st.subheader("📥 Export Chat")
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "stats": rag_pipeline.get_pipeline_stats() if rag_pipeline else {},
        }

        def convert_np(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            return str(obj)

        chat_json = json.dumps(chat_data, indent=2, default=convert_np)
        st.download_button(
            label="Download Chat History",
            data=chat_json,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

###### CHAT INPUT - BOTTOM
prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)
    with st.spinner("🤔 Thinking..."):
        try:
            response_data = rag_pipeline.answer_question(
                prompt,
                include_sources=st.session_state.get('show_sources', True)
            )
            answer = response_data.get('answer', "Sorry, no answer could be produced.")
            sources = response_data.get('sources', [])
            confidence = response_data.get('confidence')
            # Streaming response
            if st.session_state.get('enable_streaming', True) and isinstance(answer, str):
                stream_placeholder = st.empty()
                displayed_text = ""
                for char in answer:
                    displayed_text += char
                    stream_placeholder.markdown(
                        f"<div style='background-color:#f3f9eb;padding:8px 14px;border-radius:7px;margin-top:8px;font-family:inherit'><b>Assistant:</b> {displayed_text}▌</div>",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)
                stream_placeholder.empty()
            display_message(
                "assistant",
                answer,
                sources[:st.session_state.get('max_sources', 3)] if st.session_state.get('show_sources', True) else None,
                confidence
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence
            })
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error. Please try again.",
                "sources": [],
                "confidence": 0.0
            })
