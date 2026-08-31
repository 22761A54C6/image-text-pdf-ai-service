import time

import requests
from pymongo import UpdateOne

from app.config import CATEGORIES_API_URL, GEMINI_EMBED_DIMENSIONS
from app.database import db
from app.embeddings import get_embeddings_batch

# How many categories to embed+upsert per group. Kept separate from
# EMBED_BATCH_SIZE (the per-Gemini-call size) -- this controls how much
# work happens between pauses, so a quota hit on one group doesn't lose
# progress already made on earlier groups.
SYNC_CHUNK_SIZE = 50
SYNC_CHUNK_DELAY_SECONDS = 5  # pause between chunks to ease quota pressure


def fetch_real_categories():
    response = requests.get(
        CATEGORIES_API_URL,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    response.raise_for_status()
    return response.json()


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def sync_categories():
    """Raises on failure -- callers decide whether to swallow (startup)
    or surface the error (the /sync/categories endpoint)."""
    categories = fetch_real_categories()
    print(f"[category_sync] Fetched {len(categories)} categories from Spring Boot API")

    # Only sync top-level categories -- items with a non-null parentCategoryId
    # are subcategories nested under another category (e.g. "Fried Rice" under
    # "Rice & Pulao") and would otherwise pollute the vector index with
    # near-duplicate/overlapping entries.
    top_level = [
        c for c in categories
        if isinstance(c, str) or c.get("parentCategoryId") is None
    ]
    skipped = len(categories) - len(top_level)
    print(f"[category_sync] {len(top_level)} top-level categories "
          f"(skipped {skipped} with a non-null parentCategoryId)")

    normalized = [
        {"name": c, "sourceId": None} if isinstance(c, str)
        else {"name": c.get("name"), "sourceId": c.get("id")}
        for c in top_level
    ]

    # Filter out categories with empty or None names to avoid Gemini embedding errors
    normalized = [c for c in normalized if c["name"] and c["name"].strip()]

    if not normalized:
        print("[category_sync] no categories to process, aborting")
        return

    collection = db["categories"]

    # Don't re-embed categories that already exist with the same name --
    # only new categories or ones whose name actually changed need a
    # fresh Gemini call. This is what was burning quota on every sync:
    # re-embedding the same 65 unchanged categories every single run.
    existing_docs = {
        doc.get("sourceId") if doc.get("sourceId") is not None else doc.get("name"): doc
        for doc in collection.find({}, {"sourceId": 1, "name": 1, "embedding": 1})
    }

    to_embed = []
    unchanged = []
    for record in normalized:
        key = record["sourceId"] if record["sourceId"] is not None else record["name"]
        existing = existing_docs.get(key)
        if existing and existing.get("name") == record["name"] and existing.get("embedding"):
            unchanged.append(record)
        else:
            to_embed.append(record)

    print(f"[category_sync] {len(unchanged)} unchanged (skipping embedding), "
          f"{len(to_embed)} new/changed (will embed)")

    chunks = list(_chunked(to_embed, SYNC_CHUNK_SIZE))
    total_written = 0

    if not chunks:
        print("[category_sync] nothing new to embed")
    else:
        print(f"[category_sync] processing {len(to_embed)} categories in "
              f"{len(chunks)} chunk(s) of up to {SYNC_CHUNK_SIZE}")

    for chunk_num, chunk in enumerate(chunks, start=1):
        names = [c["name"] for c in chunk]
        print(f"[category_sync] chunk {chunk_num}/{len(chunks)}: embedding {len(names)} categories")

        embeddings = get_embeddings_batch(names, input_type="document")

        if len(embeddings) != len(chunk):
            raise RuntimeError(
                f"embedding count mismatch on chunk {chunk_num}: "
                f"{len(embeddings)} embeddings for {len(chunk)} categories"
            )

        operations = []
        for record, embedding in zip(chunk, embeddings):
            doc = dict(record)
            doc.pop("_id", None)
            doc["embeddedText"] = record["name"]
            doc["embedding"] = embedding

            record_id = record.get("sourceId")
            filter_ = {"sourceId": record_id} if record_id is not None else {"name": record["name"]}
            operations.append(UpdateOne(filter_, {"$set": doc}, upsert=True))

        result = collection.bulk_write(operations)
        chunk_written = result.upserted_count + result.modified_count
        total_written += chunk_written
        print(f"[category_sync] chunk {chunk_num}/{len(chunks)}: upserted {chunk_written} docs")

        if chunk_num < len(chunks):
            time.sleep(SYNC_CHUNK_DELAY_SECONDS)

    print(f"[category_sync] upserted {total_written} docs into 'categories' total "
          f"({len(unchanged)} left untouched, already up to date)")

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

    print(f"[category_sync] DONE -- {total_written} categories live, searchable via 'category_vector_index'")