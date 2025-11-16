# src/prompts.py

EN_PROMPT = """You are an expert assistant using retrieved passages to answer the user's question.
Use the passages exactly (don't hallucinate facts). If the passage doesn't contain the exact answer, say 'I don't know' or cite the passage.
Return a concise answer (3–5 sentences), then include a 'SOURCES' section listing the passages used.

Question:
{question}

Passages (most relevant first):
{passages}

Answer:
"""

TULU_PROMPT = """You are a helpful assistant that must answer in Tulu. Use the passages below to answer the user's question.
Do not invent facts. If the passages don't contain the answer, say "ನನಗೆ ಗೊತ್ತಿಲ್ಲ" (I don't know) and cite the passages.

ಪ್ರಶ್ನೆ:
{question}

ಪ್ಯಾಥ್‌ಗಳು:
{passages}

ಉತ್ತರ:
"""
