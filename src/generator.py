# src/generator.py
import os
from typing import List
from openai import OpenAI

# Don't initialize client at import time - do it lazily
_client = None

def _get_client():
    """Get or create OpenAI client"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file or environment variable.\n"
                "Example: $env:OPENAI_API_KEY='your-key-here'"
            )
        _client = OpenAI(api_key=api_key)
    return _client

def generate_openai(prompt: str, max_tokens=256, temperature=0.2):
    """Generate text using OpenAI API (v1.0+ compatible)"""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        
        # Check for quota error
        if "429" in error_msg or "quota" in error_msg.lower():
            # Extract question from prompt if possible
            return """⚠️ OpenAI API quota exceeded. 

✅ **Good news:** You can still use mBART or mT5 translation models!
- Select "mBART" or "mT5" in the sidebar
- Click "Initialize/Update Pipeline"  
- These models work offline without API costs

The retrieved passages below contain the information to answer your question."""
        
        # Other errors
        return f"⚠️ API Error: {error_msg[:200]}\n\n💡 Try using mBART or mT5 models instead (select in sidebar)."


def generate_simple_answer(passages: List[str], question: str) -> str:
    """
    Generate a simple answer without using OpenAI API.
    Just summarizes the top passages.
    """
    if not passages:
        return "No relevant passages found to answer the question."
    
    answer = f"Based on the retrieved passages:\n\n"
    
    # Take top 3 passages
    for i, passage in enumerate(passages[:3], 1):
        # Take first 200 chars of each passage
        snippet = passage[:200].strip()
        if len(passage) > 200:
            snippet += "..."
        answer += f"{i}. {snippet}\n\n"
    
    answer += "\n💡 **Note:** Using simple passage extraction. For better answers with OpenAI, add API credits or use mBART/mT5 for translation."
    
    return answer
