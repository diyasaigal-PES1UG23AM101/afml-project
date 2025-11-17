# src/retriever.py
import json
import os
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Use robust paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "data/processed/faiss_index.bin")
DATA_PATH = os.path.join(ROOT, "data/processed/all_passages.jsonl")

EMBED_MODEL = "sentence-transformers/LaBSE"

@st.cache_resource
def load_retriever_components():
    """
    Loads all necessary components for retrieval.
    Using st.cache_resource ensures this runs only once.
    """
    print("--- Loading retriever model, index, and data ---")
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    texts = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print("--- Retriever components loaded successfully ---")
    return model, index, texts

def retrieve(query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
    """
    Returns list of (idx, score, passage_text) sorted by score.
    """
    model, index, texts = load_retriever_components() # This will be instant after the first run
    
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec.astype('float32'), top_k)
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        results.append((int(idx), float(dist), texts[idx]))
    return results
