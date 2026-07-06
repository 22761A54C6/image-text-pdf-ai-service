import re
import json
from typing import List
import requests
from fastapi import HTTPException
from app.config import OLLAMA_URL, OLLAMA_MODEL


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

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
                "num_ctx": 4096
            }
        }, timeout=180)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {e}")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="LLM extraction failed")

    try:
        text_out = response.json().get("response", "").strip()
    except ValueError:
        raise HTTPException(status_code=502, detail="Ollama returned non-JSON response")

    match = re.search(r"\[.*\]", text_out, re.DOTALL)
    if not match:
        raise HTTPException(status_code=502, detail=f"No JSON array found in LLM output: {text_out[:300]}")

    try:
        products = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {text_out[:300]}")

    return products