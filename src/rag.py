from src.retriever import retrieve
from src.reranker import rank
from src.prompts import EN_PROMPT, TULU_PROMPT
from src.generator import generate_openai

def format_passages(ranked, top_n=5):
    out = []
    for i,(idx,score,txt) in enumerate(ranked[:top_n]):
        out.append(f"[{idx}] {txt[:300].strip()}")
    return "\n\n".join(out)

def rag_answer(question, lang="en", top_k=10, rerank=True):
    # 1) retrieve
    raw = retrieve(question, top_k)
    if not raw:
        return "No passages found."

    if rerank:
        candidates = [p for (_,_,p) in raw]
        ranked_idxscore = rank(question, candidates)
        # ranked_idxscore returns list of (candidate_index, score)
        # convert to (orig_idx, score, text)
        ranked = []
        for cand_idx, s in ranked_idxscore:
            orig_idx, dist, text = raw[cand_idx]
            ranked.append((orig_idx, s, text))
    else:
        ranked = raw

    passages_text = format_passages(ranked, top_n=5)

    if lang == "en":
        prompt = EN_PROMPT.format(question=question, passages=passages_text)
        answer = generate_openai(prompt)
        return answer, ranked[:5]
    elif lang == "tcy" or lang == "tulu":
        prompt = TULU_PROMPT.format(question=question, passages=passages_text)
        answer = generate_openai(prompt)
        return answer, ranked[:5]
    elif lang == "both":
        combined = "Please answer first in English, then translate to Tulu. Use these passages to support your answer.\n\n" + EN_PROMPT.format(question=question, passages=passages_text)
        answer = generate_openai(combined)
        return answer, ranked[:5]
    else:
        # fallback: english
        prompt = EN_PROMPT.format(question=question, passages=passages_text)
        answer = generate_openai(prompt)
        return answer, ranked[:5]
