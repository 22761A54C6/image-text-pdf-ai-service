import re
import json
from typing import List
import requests
from fastapi import HTTPException
from app.config import GROQ_URL, GROQ_MODEL, GROQ_API_KEY


def extract_products_with_llm(raw_text: str) -> List[dict]:
    prompt = f"""You are a menu parser. Extract items from this OCR text.
OCR TEXT:
{raw_text}
Return ONLY a JSON array, no explanation, no markdown fences. Each item:
{{"name": "...", "price": number}}
IMPORTANT RULES:
- All output text (name) MUST be in English only. Never use Chinese or any other language.
- If price is missing for a line, skip that line entirely.
- Do not repeat the same item twice.
Extract every distinct product name and price you see in the text/image.
Do not skip, merge, or omit any item, even if the name or price looks 
garbled or uncertain.
Before finalizing, count the number of items in your output and the 
number of distinct product-like lines in the source. If they don't 
match, go back and find what's missing.
"""

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            timeout=60,  # Groq is fast — no need for Ollama's long 180s timeout
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Groq: {e}")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq extraction failed: {response.text[:300]}")

    try:
        text_out = response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Groq returned unexpected response format")

    match = re.search(r"\[.*\]", text_out, re.DOTALL)
    if not match:
        raise HTTPException(status_code=502, detail=f"No JSON array found in LLM output: {text_out[:300]}")

    try:
        products = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {text_out[:300]}")

    return products