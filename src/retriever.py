# src/retriever.py
import json, os
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(ROOT, "data/processed/faiss_index.bin")
DATA_PATH = os.path.join(ROOT, "data/processed/all_passages.jsonl")

# IMPORTANT: use the same model you used to build index
EMBED_MODEL = "sentence-transformers/LaBSE"

_model = None
_index = None
_texts = None

def _ensure_loaded():
    global _model, _index, _texts
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _texts is None:
        _texts = []
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                _texts.append(json.loads(line)["text"])
    return _model, _index, _texts

def retrieve(query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
    """
    Returns list of (idx, score, passage_text) sorted by score (smaller L2 distance is better).
    """
    model, index, texts = _ensure_loaded()
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        results.append((int(idx), float(dist), texts[idx]))
    return results
