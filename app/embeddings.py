
from typing import List

import requests

from app.config import OLLAMA_EMBED_URL, OLLAMA_EMBED_MODEL
from app.database import db
from app.matching import find_confident_category


def get_embedding(text: str) -> List[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    return response.json()["embedding"]


def embed_and_store(products: List[dict]):
    for product in products:
        try:
            embedding = get_embedding(product["name"])

            match = find_confident_category(embedding)

            db.products.update_one(
                {"name": product["name"]},
                {"$set": {
                    "name": product["name"],
                    "price": product.get("price"),
                    "embedding": embedding,
                    "matchedCategory": match["name"] if match else None,
                    "matchedCategoryScore": match["score"] if match else None,
                    "needsReview": match is None,
                }},
                upsert=True
            )
        except Exception as e:
            print(f"[embed_and_store] failed for '{product.get('name')}': {e}")