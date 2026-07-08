import json

from google.genai.types import GenerateContentConfig
from google import genai

from app.config import GEMINI_API_KEY, GEMINI_TEXT_MODEL

_client = genai.Client()

_SYSTEM_PROMPT = (
    "You normalize restaurant menu item names for search matching. "
    "You will receive a JSON array of raw product names. "
    "Return ONLY a JSON array of the same length, in the same order, with each "
    "name normalized as plain text -- no preamble, no explanation, no markdown, no code.\n\n"
    "Rules per name:\n"
    "- lowercase everything\n"
    "- remove extra/duplicate spaces\n"
    "- remove punctuation\n"
    "- remove duplicate words\n"
    "- remove emojis\n"
    "- remove portion markers like '(Full)', '(Half)', 'full', 'half'\n"
    "- remove ANY parenthetical notes, tags, or asides that are not part of the actual "
    "dish name -- e.g. '(Check Availability)', '(Seasonal)', '(Ask Staff)', '(Limited)', "
    "'(New)' -- these are restaurant remarks, not the food item, and must be dropped entirely\n"
    "- fix OCR/spelling mistakes and normalize spelling variants to one canonical form "
    "(e.g. 'Chicken Biriyani', 'Chicken Dum Biriyani' -> 'chicken dum biryani')\n"
    "- keep the dish identity intact -- don't drop meaningful words like 'dum', 'tandoori', 'boneless', etc.\n\n"
    "Example:\n"
    "Input: [\"Chicken Dum Biriyani (Full)\", \"CHICKEN KUSKA (CHECK AVAILABILITY)\", \"Cup Cake\"]\n"
    "Output: [\"chicken dum biryani\", \"chicken kuska\", \"cupcake\"]"
)


def normalize_product_names_batch(raw_names: list[str]) -> list[str]:
    """Single Gemini call for the whole batch -- keeps quota usage to
    ~1 request regardless of how many product names are in the menu."""
    if not raw_names:
        return []

    if not GEMINI_API_KEY:
        print("[normalize] GEMINI_API_KEY not set, falling back to basic cleanup")
        return [_basic_fallback(n) for n in raw_names]

    try:
        response = _client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=f"{_SYSTEM_PROMPT}\n\nInput: {json.dumps(raw_names, ensure_ascii=False)}\nOutput:",
            config=GenerateContentConfig(temperature=0.1),
        )
        text_out = (response.text or "").strip()

        # Strip accidental code fences if the model adds them anyway
        if text_out.startswith("```"):
            text_out = text_out.strip("`")
            if text_out.startswith("json"):
                text_out = text_out[4:].strip()

        normalized = json.loads(text_out)

        if not isinstance(normalized, list) or len(normalized) != len(raw_names):
            print(f"[normalize] batch result mismatch ({len(normalized) if isinstance(normalized, list) else 'not a list'} "
                  f"vs {len(raw_names)} expected), falling back for all")
            return [_basic_fallback(n) for n in raw_names]

        # Per-item safety net: any malformed entry (code, multi-line, too long) falls back individually
        result = []
        for raw, norm in zip(raw_names, normalized):
            if not isinstance(norm, str) or not norm.strip() or "```" in norm or "\n" in norm or len(norm) > 100:
                result.append(_basic_fallback(raw))
            else:
                result.append(norm.strip())
        return result

    except Exception as e:
        print(f"[normalize] Gemini batch normalization failed: {e}, falling back for all")
        return [_basic_fallback(n) for n in raw_names]


def _basic_fallback(raw_name: str) -> str:
    import re
    text = raw_name.lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\bfull\b|\bhalf\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " ".join(seen).strip()