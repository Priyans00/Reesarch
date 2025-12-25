"""
Research Paper Question Answering System

A RAG-based system for answering questions about research papers.
Uses efficient small models for each task:
- Embedding: allenai-specter (scientific paper embeddings)
- Reranking: MS-MARCO MiniLM-L6-v2 (cross-encoder)
- Generation: Qwen2.5-0.5B-Instruct (lightweight LLM)

Run with: streamlit run app.py
"""
import os
import sys
import tempfile
from pathlib import Path
import traceback

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline, create_pipeline
from config import UPLOADS_DIR


# Page configuration
st.set_page_config(
    page_title="Research Paper QA",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached)."""
    return create_pipeline(auto_load=True)


def process_uploaded_files(pipeline, uploaded_files, status_container):
    """Process and index uploaded files."""
    progress_bar = status_container.progress(0)
    status_text = status_container.empty()
    
    success_count = 0
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.processed_files:
            status_text.text(f"Skipping (already processed): {uploaded_file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))
            continue
            
        status_text.text(f"Processing: {uploaded_file.name}")
        
        try:
            # Save to uploads directory
            file_path = UPLOADS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text(f"Indexing: {uploaded_file.name}")
            
            # Ingest and index
            pipeline.add_document(str(file_path))
            st.session_state.processed_files.add(uploaded_file.name)
            success_count += 1
            status_container.success(f"✓ Indexed: {uploaded_file.name}")
            
        except Exception as e:
            status_container.error(f"✗ Error with {uploaded_file.name}: {str(e)}")
            traceback.print_exc()
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text(f"Done! Processed {success_count} files.")
    return success_count


def main():
    st.title("📚 Research Paper Question Answering")
    st.markdown("""
    Upload research papers and ask questions about their content.
    The system uses RAG (Retrieval-Augmented Generation) to provide accurate, 
    context-grounded answers.
    """)
    
    # Initialize pipeline
    try:
        with st.spinner("Loading models..."):
            pipeline = load_pipeline()
        st.session_state.pipeline_ready = True
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        traceback.print_exc()
        return
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        # Status container for processing messages
        status_container = st.container()
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Process button
            if st.button("🚀 Process & Index Documents", type="primary", key="process_btn"):
                with status_container:
                    process_uploaded_files(pipeline, uploaded_files, status_container)
        
        st.divider()
        
        # Show indexed documents
        st.subheader("Indexed Documents")
        stats = pipeline.get_statistics()
        
        if stats["total_documents"] > 0:
            for doc in pipeline.list_documents():
                with st.expander(f"📄 {doc['filename'][:30]}..."):
                    st.write(f"**Title:** {doc['title'][:100]}")
                    st.write(f"**Pages:** {doc['num_pages']}")
                    st.write(f"**ID:** {doc['doc_id']}")
        else:
            st.info("No documents indexed yet. Upload PDFs above.")
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["total_documents"])
        with col2:
            st.metric("Chunks", stats["total_chunks"])
        
        # Clear index button
        if st.button("🗑️ Clear Index"):
            pipeline.clear_index()
            st.rerun()
    
    # Main content area
    st.header("💬 Ask Questions")
    
    # Query input
    question = st.text_input(
        "Enter your question about the research papers:",
        placeholder="What is the main contribution of the paper?"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of sources", 1, 10, 5)
        with col2:
            use_reranking = st.checkbox("Use reranking", value=True)
    
    # Query button
    if st.button("🔍 Get Answer", type="primary", disabled=not question):
        if stats["total_chunks"] == 0:
            st.warning("Please upload and index documents first.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                result = pipeline.query(
                    question=question,
                    top_k=top_k,
                    use_reranking=use_reranking
                )
            
            # Display answer
            st.subheader("📝 Answer")
            
            if result.abstained:
                st.warning(result.answer)
            else:
                st.markdown(result.answer)
            
            # Confidence indicator
            confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** :{confidence_color}[{result.confidence:.0%}]")
            
            # Sources
            if result.sources:
                st.subheader("📚 Sources")
                for i, source in enumerate(result.sources, 1):
                    with st.expander(f"Source {i} - {source['section'].upper()}"):
                        st.markdown(f"**Document ID:** {source['doc_id']}")
                        st.markdown(f"**Section:** {source['section']}")
                        st.markdown("**Preview:**")
                        st.text(source['text_preview'])
            
            # Show raw context (collapsed)
            with st.expander("🔍 View Retrieved Context"):
                st.text(result.context)
    
    # Example questions
    st.divider()
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "What is the main contribution of this paper?",
        "What methods were used in the experiments?",
        "What were the key findings and results?",
        "What are the limitations of the proposed approach?",
        "How does this work compare to previous methods?",
        "What datasets were used for evaluation?",
    ]
    
    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}"):
                st.session_state.question = q
                st.rerun()


if __name__ == "__main__":
    main()
