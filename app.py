# app/app.py
# IMPORTANT: Load environment variables FIRST before any other imports
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Tulu RAG Demo", page_icon="🗣️", layout="wide")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

st.title("🗣️ Tulu RAG — Ask in English or Tulu")
st.markdown("*Retrieval-Augmented Generation system for Tulu language question answering*")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    translation_model = st.selectbox(
        "Translation Model",
        ["GPT-4 (API)", "mBART", "mT5"],
        help="Choose the translation model for Tulu responses"
    )
    
    use_reranking = st.checkbox("Use Re-ranking", value=True, help="Re-rank retrieved passages for better relevance")
    
    top_k = st.slider("Passages to retrieve", min_value=5, max_value=20, value=10, help="Number of passages to retrieve")
    
    rerank_top = st.slider("Top passages to use", min_value=3, max_value=10, value=5, help="Number of top passages after re-ranking")
    
    # Initialize pipeline button
    if st.button("Initialize/Update Pipeline"):
        model_map = {"GPT-4 (API)": "gpt4", "mBART": "mbart", "mT5": "mt5"}
        selected_model = model_map[translation_model]
        
        if selected_model in ["mbart", "mt5"]:
            st.warning("⏳ **First-time download:** This will take 2-5 minutes to download the model (~1-2GB). Please be patient!")
            st.info("💡 The model is being cached locally and will load instantly next time.")
        
        with st.spinner(f"Loading {translation_model} model... (check terminal for progress)"):
            try:
                st.session_state.pipeline = RAGPipeline(
                    translation_model=selected_model,
                    use_reranking=use_reranking
                )
                st.success(f"✅ Pipeline initialized with {translation_model}!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.info("Try selecting a different model or check your internet connection.")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    This RAG system:
    - Retrieves relevant passages from Tulu corpus
    - Re-ranks for relevance
    - Generates contextual answers
    - Supports multilingual responses
    """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    q = st.text_input("❓ Your Question", "", placeholder="e.g., What is the history of Tulu language?")

with col2:
    lang = st.radio("Response Language", ("English", "Tulu", "Both"))

if st.button("🔍 Ask", type="primary") and q.strip():
    # Check if pipeline is initialized
    if st.session_state.pipeline is None:
        st.warning("⚠️ Please initialize the pipeline first using the sidebar.")
    else:
        # Check if FAISS index exists
        index_path = "data/processed/faiss_index.bin"
        if not os.path.exists(index_path):
            st.error(f"❌ FAISS index not found at `{index_path}`")
            st.info("""
            **To build the index, you need to:**
            1. Collect data: `python diya/src/scrape_wiki.py --limit 50`
            2. Build dataset: `python src/build_full_dataset.py`
            3. Create index: `python src/embed_index.py`
            
            **Or for testing, you can create a demo index with sample data.**
            """)
            st.stop()
        
        with st.spinner("Retrieving and generating answer..."):
            try:
                lang_code = {"English": "en", "Tulu": "tulu", "Both": "both"}[lang]
                
                result = st.session_state.pipeline.query(
                    question=q,
                    response_language=lang_code,
                    top_k=top_k,
                    rerank_top=rerank_top
                )
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])
                
                # Display retrieved passages
                st.markdown("### 📚 Retrieved Passages")
                st.caption(f"Top {len(result['passages'])} most relevant passages")
                
                for i, (idx, score, txt) in enumerate(result["passages"], 1):
                    with st.expander(f"Passage {i} — ID: {idx} (Relevance: {score:.3f})"):
                        st.write(txt[:800])
                        if len(txt) > 800:
                            st.caption("... (truncated)")
                
                # Display metadata
                with st.expander("ℹ️ Query Details"):
                    st.json({
                        "Original Question": result["question"],
                        "Search Query": result["search_query"],
                        "Response Language": result["language"],
                        "Passages Retrieved": result["num_passages"]
                    })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Tulu RAG System | AFML Project | PES University</small>
</div>
""", unsafe_allow_html=True)
