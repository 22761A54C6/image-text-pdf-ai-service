import re
import json
import time
from typing import List

from fastapi import HTTPException
from google import genai
from google.genai.types import GenerateContentConfig

from app.config import GEMINI_API_KEY, GEMINI_TEXT_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def extract_products_with_llm(raw_text: str) -> List[dict]:
    prompt = f"""You are a menu parser. Extract items from this OCR text.
OCR TEXT:
{raw_text}
Return ONLY a JSON array, no explanation, no markdown fences. Each item:
{{"name": "...", "price": number or null}}
IMPORTANT RULES:
- All output text (name) MUST be in English only. Never use Chinese or any other language.
- If price is missing for a line, include the item with price set to null.
- Do not repeat the same item twice.
Extract every distinct product name you see in the text/image.
Do not skip, merge, or omit any item, even if the name looks garbled or uncertain.
Before finalizing, count the number of items in your output and the 
number of distinct product-like lines in the source. If they don't 
match, go back and find what's missing.
"""

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    response = None
    last_error = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=prompt,
                config=GenerateContentConfig(temperature=0.1, max_output_tokens=8192),
            )
            break  # success, stop retrying
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[extract] Gemini overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait}s")
                time.sleep(wait)
                continue
            # Non-retryable error -- fail immediately
            raise HTTPException(status_code=502, detail=f"Could not reach Gemini: {e}")

    if response is None:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini is still unavailable after {max_retries} retries: {last_error}"
        )

    text_out = (response.text or "").strip()

    finish_reason = None
    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError):
        pass

    if finish_reason == "MAX_TOKENS":
        print(f"[extract] WARNING: Gemini hit max_output_tokens, response was truncated. "
              f"Attempting to salvage partial output.")

    if not text_out:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response")

    match = re.search(r"\[.*\]", text_out, re.DOTALL)
    if match:
        try:
            products = json.loads(match.group(0))
            return products
        except json.JSONDecodeError:
            pass

    partial_objects = re.findall(r"\{[^{}]*\}", text_out)
    if partial_objects:
        products = []
        for obj in partial_objects:
            try:
                products.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if products:
            print(f"[extract] WARNING: recovered {len(products)} products from truncated/malformed output")
            return products

    raise HTTPException(status_code=502, detail=f"No JSON array found in LLM output: {text_out[:300]}")