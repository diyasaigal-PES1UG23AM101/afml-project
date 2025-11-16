# src/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Tuple

CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small & fast

_ce = None
def rank(query: str, candidates: List[str]) -> List[Tuple[int, float]]:
    """
    Input: textual candidates list.
    Output: list of (candidate_index, score) descending (higher = better).
    """
    global _ce
    if _ce is None:
        _ce = CrossEncoder(CROSS_MODEL)
    texts = [[query, c] for c in candidates]
    scores = _ce.predict(texts)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed
