# app/app.py

import streamlit as st
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline import run_rag_pipeline

st.set_page_config(page_title="Tulu Language RAG System", page_icon="🌍", layout="wide")

st.title("🌍 Tulu Language Preservation RAG System")
st.markdown("This app uses a RAG system to answer questions. Ask in English or Tulu.")

with st.sidebar:
    st.header("Controls")
    # Updated language options
    language_options = {
        "English": "en",
        "Tulu": "tulu",
        "Both (English + Tulu)": "both"
    }
    selected_language_name = st.radio(
        "Select Language:",
        options=list(language_options.keys()),
    )
    language_code = language_options[selected_language_name]

    st.subheader("Pipeline Settings")
    retriever_top_k = st.slider("Retriever Top-K", 5, 50, 20)
    reranker_top_k = st.slider("Reranker Top-K", 1, 10, 5)

query = st.text_input("Enter your question:", "")

if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding answer..."):
            try:
                result = run_rag_pipeline(
                    query,
                    language=language_code,
                    retriever_top_k=retriever_top_k,
                    reranker_top_k=reranker_top_k
                )
                st.subheader("Answer:")
                st.markdown(result["answer"])
                with st.expander("Show Sources"):
                    for i, source in enumerate(result["sources"]):
                        st.info(f"Source {i+1}:\n{source}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please ensure your API key is set and data/index files exist.")
