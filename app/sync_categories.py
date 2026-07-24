import requests
from pymongo import UpdateOne

from app.config import CATEGORIES_API_URL, GEMINI_EMBED_DIMENSIONS
from app.database import db
from app.embeddings import get_embeddings_batch


def fetch_real_categories():
    response = requests.get(
        CATEGORIES_API_URL,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    response.raise_for_status()
    return response.json()


def sync_categories():
    """Raises on failure -- callers decide whether to swallow (startup)
    or surface the error (the /sync/categories endpoint)."""
    categories = fetch_real_categories()
    print(f"[category_sync] Fetched {len(categories)} categories from Spring Boot API")

    normalized = [
        {"name": c, "sourceId": None} if isinstance(c, str)
        else {"name": c.get("name"), "sourceId": c.get("id")}
        for c in categories
    ]

    # Filter out categories with empty or None names to avoid Gemini embedding errors
    normalized = [c for c in normalized if c["name"] and c["name"].strip()]

    if not normalized:
        print("[category_sync] no categories to process, aborting")
        return

    names = [c["name"] for c in normalized]
    embeddings = get_embeddings_batch(names, input_type="document")
    print(f"[category_sync] {len(embeddings)} vectors generated via Gemini")

    if len(embeddings) != len(normalized):
        raise RuntimeError(
            f"embedding count mismatch: {len(embeddings)} embeddings for {len(normalized)} categories"
        )

    collection = db["categories"]
    operations = []
    for record, embedding in zip(normalized, embeddings):
        doc = dict(record)
        doc.pop("_id", None)
        doc["embeddedText"] = record["name"]
        doc["embedding"] = embedding

        record_id = record.get("sourceId")
        filter_ = {"sourceId": record_id} if record_id is not None else {"name": record["name"]}
        operations.append(UpdateOne(filter_, {"$set": doc}, upsert=True))

    result = collection.bulk_write(operations)
    written = result.upserted_count + result.modified_count
    print(f"[category_sync] upserted {written} docs into 'categories'")

    # --- remove categories that no longer exist upstream ---
    valid_source_ids = {r["sourceId"] for r in normalized if r["sourceId"] is not None}
    valid_names = {r["name"] for r in normalized if r["sourceId"] is None}

    delete_result = collection.delete_many({
        "$or": [
            {"sourceId": {"$ne": None, "$nin": list(valid_source_ids)}},
            {"sourceId": None, "name": {"$nin": list(valid_names)}}
        ]
    })
    print(f"[category_sync] removed {delete_result.deleted_count} stale categories")

    try:
        existing = {idx["name"] for idx in collection.list_search_indexes()}
    except Exception as e:
        # Index listing can fail on some Atlas tiers/permissions -- not fatal,
        # the category data sync above already succeeded regardless.
        print(f"[category_sync] could not list search indexes: {e}")
        existing = set()

    if "category_vector_index" in existing:
        print("[category_sync] index 'category_vector_index' already exists, skipping")
    else:
        try:
            collection.create_search_index({
                "name": "category_vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": GEMINI_EMBED_DIMENSIONS, "similarity": "cosine"}
                    ]
                },
            })
            print(f"[category_sync] created index 'category_vector_index' ({GEMINI_EMBED_DIMENSIONS} dims). "
                  f"Atlas needs ~1-2 min to finish building it.")
        except Exception as e:
            # Most likely the index already exists (race/listing failed above) --
            # log and continue rather than fail the whole sync over this.
            print(f"[category_sync] index creation skipped/failed (may already exist): {e}")

    print(f"[category_sync] DONE -- {written} categories live, searchable via 'category_vector_index'")