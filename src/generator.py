# src/generator.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# The client will now find the GROQ_API_KEY from your .env file
client = Groq()

def generate_openai(prompt: str, max_tokens=256, temperature=0.1):
    #Generates a response using the Groq API.
    response = client.chat.completions.create(
        # We use a powerful open-source model available on Groq
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()
