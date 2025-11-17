# src/reranker.py
import streamlit as st
from sentence_transformers import CrossEncoder
from typing import List, Tuple

CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@st.cache_resource
def get_reranker_model():
    """
    Loads and caches the cross-encoder model.
    """
    print("--- Loading reranker model ---")
    model = CrossEncoder(CROSS_MODEL)
    print("--- Reranker model loaded successfully ---")
    return model

def rank(query: str, candidates: List[str]) -> List[Tuple[int, float]]:
    """
    Input: textual candidates list.
    Output: list of (candidate_index, score) descending.
    """
    model = get_reranker_model() # This will be instant after the first run
    
    texts = [[query, c] for c in candidates]
    scores = model.predict(texts, show_progress_bar=False) # Progress bar is not needed here
    
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed
