import re
import json
from typing import List
import requests
from fastapi import HTTPException
from app.config import OLLAMA_URL, OLLAMA_MODEL

JUNK_LINES = {
    "lorem ipsum", "menucard", "menu card", "menu", "getty images", "getty",
}


def is_junk_line(line: str) -> bool:
    lower = line.lower().strip()
    if lower in JUNK_LINES:
        return True
    if line.strip().isdigit() and len(line.strip()) <= 3:
        return True
    if re.fullmatch(r"[\d\s/\-()]{7,}", line.strip()):
        return True
    return False


def clean_name(name: str) -> str:
    name = re.sub(r"[.\-·•_]{2,}\s*$", "", name).strip()
    name = name.rstrip(".").strip()
    # Strip menu-card noise phrases that pollute embeddings
    name = re.sub(r"\(\s*check\s+availability\s*\)", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\(\s*subject\s+to\s+availability\s*\)", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def extract_products_regex(raw_text: str) -> List[dict]:
    """Handles two common menu layouts:
    1. Name and price on the SAME line ('Chicken Biryani 250', 'Pizza - $12.99')
    2. Name on one line, price on the NEXT line (common in stylized templates)
    """
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    products = []
    seen = set()

    def try_add(name: str, price_str: str):
        name = clean_name(name)
        if not name or len(name) < 2 or is_junk_line(name):
            return False
        try:
            price = float(price_str.replace(",", "."))
        except ValueError:
            return False
        key = name.lower()
        if key in seen:
            return False
        seen.add(key)
        products.append({"name": name, "price": price})
        return True

    i = 0
    while i < len(lines):
        line = lines[i]
        if is_junk_line(line):
            i += 1
            continue
        same_line_match = re.search(
            r"^(.*?[a-zA-Z].*?)[\s.\-·•_]*(?:₹|rs\.?|inr|\$|€|£)?\s?(\d{1,6}(?:[.,]\d{1,2})?)\s*(?:/-)?$",
            line, re.IGNORECASE
        )
        if same_line_match:
            name_part = same_line_match.group(1).strip()
            price_part = same_line_match.group(2)
            if try_add(name_part, price_part):
                i += 1
                continue
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            price_only_match = re.fullmatch(
                r"(?:₹|rs\.?|inr|\$|€|£)?\s?(\d{1,6}(?:[.,]\d{1,2})?)\s*(?:/-)?",
                next_line, re.IGNORECASE
            )
            has_letters = re.search(r"[a-zA-Z]", line)
            if price_only_match and has_letters and not re.search(r"\d{3,}", line):
                if try_add(line, price_only_match.group(1)):
                    i += 2
                    continue
        i += 1
    return products


def extract_products_with_llm(raw_text: str) -> List[dict]:
    """Fallback only — used when regex extraction finds nothing (unusual menu layout)."""
    prompt = f"""You are a menu parser. Extract items from this OCR text.
OCR TEXT:
{raw_text}
Return ONLY a JSON array, no explanation, no markdown fences. Each item:
{{"name": "...", "price": number}}
IMPORTANT RULES:
- All output text (name) MUST be in English only. Never use Chinese or any other language.
- If price is missing for a line, skip that line entirely.
- Skip placeholder/filler lines like "Lorem ipsum".
- Do not repeat the same item twice.
"""
    

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



    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="LLM extraction failed")
    text_out = response.json().get("response", "").strip()
    text_out = text_out.replace("```json", "").replace("```", "").strip()
    try:
        products = json.loads(text_out)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {text_out[:300]}")
    return products