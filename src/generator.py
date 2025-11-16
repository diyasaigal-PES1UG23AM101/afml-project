# src/generator.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # set in environment

def generate_openai(prompt: str, max_tokens=256, temperature=0.2):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini", # or gpt-4o-mini or gpt-4o-mini-tts depending; use available
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp["choices"][0]["message"]["content"].strip()
