# src/pipeline.py

import sys
import os
from typing import List, Dict
from groq import Groq

# Add the root directory to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.retriever import retrieve
from src.reranker import rank
from src.generator import generate_openai # This now calls our low-temperature Groq model
from src.prompts import EN_PROMPT, TULU_PROMPT
from src.utils import clean_text

# --- Helper functions ---
def format_passages_for_prompt(passages: List[str]) -> str:
    return "\n\n---\n\n".join([f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)])

def classify_query(query: str) -> str:
    client = Groq()
    classification_prompt = f"""Classify the user query as 'factual' or 'conversational'.
    Query: "{query}"
    Classification:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0, max_tokens=10
        )
        classification = response.choices[0].message.content.strip().lower()
        return "factual" if "factual" in classification else "conversational"
    except Exception:
        return "factual"

# --- NEW SELF-CORRECTION FUNCTION ---
def check_for_unhelpful_answer(answer: str) -> bool:
    """Checks if the RAG answer is a variation of 'I don't know'."""
    client = Groq()
    check_prompt = f"""
    Does the following answer essentially say "I don't know", "I cannot find the answer", or that the provided information was not helpful?
    Answer Yes or No.

    Answer: "{answer}"
    
    Response:
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0.0, max_tokens=5
        )
        decision = response.choices[0].message.content.strip().lower()
        return "yes" in decision
    except Exception:
        return False

def run_rag_pipeline(
    query: str, language: str = "en", retriever_top_k: int = 20, reranker_top_k: int = 5
) -> Dict:
    cleaned_query = clean_text(query)
    query_type = classify_query(cleaned_query)

    if query_type == "conversational":
        print("--- Query classified as conversational. Bypassing RAG. ---")
        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": cleaned_query}]
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "sources": ["This was a conversational query."]}

    print("--- Query classified as factual. Running full RAG pipeline. ---")
    
    # Standard RAG process
    retrieved_results = retrieve(cleaned_query, top_k=retriever_top_k)
    if not retrieved_results:
        return {"answer": "Could not find any relevant passages.", "sources": []}
    
    retrieved_passages = [r[2] for r in retrieved_results]
    reranked_indices = rank(cleaned_query, retrieved_passages)
    final_passages = [retrieved_passages[i] for i, score in reranked_indices[:reranker_top_k]]
    formatted_passages_str = format_passages_for_prompt(final_passages)

    if language == "tulu":
        prompt = TULU_PROMPT.format(question=cleaned_query, passages=formatted_passages_str)
    elif language == "both":
        base_prompt = EN_PROMPT.format(question=cleaned_query, passages=formatted_passages_str)
        prompt = "First, answer in English based on the passages. Second, translate your English answer into Tulu.\n\n" + base_prompt
    else:
        prompt = EN_PROMPT.format(question=cleaned_query, passages=formatted_passages_str)
        
    # Generate the initial, grounded answer
    grounded_answer = generate_openai(prompt)

    # --- SELF-CORRECTION AND FALLBACK LOGIC ---
    if check_for_unhelpful_answer(grounded_answer):
        print("--- RAG answer was unhelpful. Falling back to general knowledge. ---")
        # The first attempt failed, so we ask the LLM to answer from its own knowledge
        fallback_prompt = f"""The previous attempt to answer the user's question using a document search failed.
        Please answer the following question based on your general knowledge.

        Question: {cleaned_query}
        """
        final_answer = generate_openai(fallback_prompt)
        return {
            "answer": final_answer,
            "sources": ["Answer generated from general knowledge after document search failed."]
        }
    # --- END OF SELF-CORRECTION LOGIC ---

    return {
        "answer": grounded_answer,
        "sources": final_passages
    }