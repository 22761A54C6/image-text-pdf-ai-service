import requests

from app.config import CATEGORIES_API_URL
from app.db_injection_pipeline import DBInjectionPipeline
from app.embeddings import backfill_embeddings


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

        normalized = [
            {"name": c, "sourceId": None} if isinstance(c, str)
            else {"name": c.get("name"), "sourceId": c.get("id")}
            for c in categories
        ]

        pipeline = DBInjectionPipeline(
            collection_name="categories",
            text_field="name",
            id_field="sourceId",
            vector_index_name="category_vector_index",  # matches matching.py -- no changes needed there
        )
        pipeline.run(source=normalized)

        print("[category_sync] Category sync complete.")
    except Exception as e:
        print(f"[category_sync] FAILED -- server will start without fresh categories: {e}")

    # Catches anything sitting in categories without an embedding --
    # e.g. docs added via mongoimport/Compass rather than the pipeline above.
    # Safe to re-run: only touches docs missing 'embedding'.
    try:
        print("[category_sync] Backfilling embeddings for any un-embedded categories...")
        backfill_embeddings(
            collection_name="categories",
            text_field="name",
            vector_index_name="category_vector_index",
        )
    except Exception as e:
        print(f"[category_sync] Backfill failed: {e}")