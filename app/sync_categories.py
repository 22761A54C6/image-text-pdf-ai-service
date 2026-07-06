import requests

from app.config import CATEGORIES_API_URL
from app.embeddings import get_embedding
from app.database import db


def fetch_real_categories():
    response = requests.get(
        CATEGORIES_API_URL,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    response.raise_for_status()
    return response.json()


def sync_categories():
    try:
        categories = fetch_real_categories()
        print(f"[category_sync] Fetched {len(categories)} categories from Spring Boot API")

        db.categories.delete_many({})

        for cat in categories:
            name = cat if isinstance(cat, str) else cat.get("name")
            if not name:
                continue

            embedding = get_embedding(name)
            db.categories.insert_one({
                "name": name,
                "sourceId": cat.get("id") if isinstance(cat, dict) else None,
                "embedding": embedding
            })
            print(f"[category_sync]   synced: {name}")

        print("[category_sync] Category sync complete.")
    except Exception as e:
        print(f"[category_sync] FAILED — server will start without fresh categories: {e}")