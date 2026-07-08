import re
import json
from typing import List

from fastapi import HTTPException
from google import genai
from google.genai.types import GenerateContentConfig

from app.config import GEMINI_API_KEY, GEMINI_TEXT_MODEL

_client = genai.Client()


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

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    try:
        response = _client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=prompt,
            config=GenerateContentConfig(temperature=0.1, max_output_tokens=8192),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini: {e}")

    text_out = (response.text or "").strip()

    # Check if Gemini stopped because it ran out of tokens rather than finishing naturally
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
            pass  # fall through to salvage attempt below

    # Salvage attempt: grab complete {...} objects even if the array wasn't
    # properly closed (e.g. truncated mid-object due to max_output_tokens).
    # This regex only matches flat objects with no nested braces, which is
    # fine for our {"name": ..., "price": ...} shape -- any final incomplete
    # object (missing closing brace) is simply skipped.
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