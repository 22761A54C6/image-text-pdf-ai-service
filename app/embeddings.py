from typing import List

import requests
from pymongo import UpdateOne

from app.config import VOYAGE_API_KEY, VOYAGE_EMBED_URL, VOYAGE_EMBED_MODEL, VOYAGE_EMBED_DIMENSIONS
from app.database import db
from app.matching import find_confident_category


def get_embedding(text: str, input_type: str = "document") -> List[float]:
    """input_type: "document" for things you store, "query" for search text.
    Voyage tunes embeddings differently per type -- getting this wrong doesn't
    error, it just quietly makes matches worse."""
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set -- add it to your .env file")

    response = requests.post(
        VOYAGE_EMBED_URL,
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={
            "input": [text],
            "model": VOYAGE_EMBED_MODEL,
            "input_type": input_type,
            "output_dimension": VOYAGE_EMBED_DIMENSIONS,
        },
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def get_embeddings_batch(texts: List[str], input_type: str = "document") -> List[List[float]]:
    """Batch version for the injection pipeline -- Voyage accepts up to 128
    inputs per call, far cheaper than one call per item."""
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set -- add it to your .env file")

    embeddings: List[List[float]] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        response = requests.post(
            VOYAGE_EMBED_URL,
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={
                "input": chunk,
                "model": VOYAGE_EMBED_MODEL,
                "input_type": input_type,
                "output_dimension": VOYAGE_EMBED_DIMENSIONS,
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        embeddings.extend(item["embedding"] for item in data)
    return embeddings


def embed_and_store(products: List[dict]):
    for product in products:
        try:
            embedding = get_embedding(product["name"], input_type="document")

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


def backfill_embeddings(
    collection_name: str,
    text_field: str = "name",
    vector_index_name: str = "vector_index",
    num_dimensions: int = VOYAGE_EMBED_DIMENSIONS,
    batch_size: int = 128,
    run_matching: bool = False,
):
    """Embeds documents already sitting in a collection without vectors --
    e.g. anything loaded via mongoimport/Compass import that skipped
    embedding. Safe to re-run: only touches docs missing 'embedding'.

    run_matching=True additionally runs find_confident_category() on each
    newly embedded doc and writes matchedCategory/matchedCategoryScore/
    needsReview -- same fields embed_and_store() sets for uploaded products.
    Use this for the products collection; leave False for categories
    (categories ARE the thing being matched against, matching them against
    themselves is meaningless)."""
    collection = db[collection_name]

    docs = list(collection.find({"embedding": {"$exists": False}}))
    print(f"[backfill] {len(docs)} documents in '{collection_name}' missing embeddings")

    if docs:
        for i in range(0, len(docs), batch_size):
            chunk = docs[i:i + batch_size]
            texts = [str(d.get(text_field, "")) for d in chunk]
            embeddings = get_embeddings_batch(texts, input_type="document")

            operations = []
            for doc, text, embedding in zip(chunk, texts, embeddings):
                update_fields = {"embedding": embedding, "embeddedText": text}

                if run_matching:
                    match = find_confident_category(embedding)
                    update_fields["matchedCategory"] = match["name"] if match else None
                    update_fields["matchedCategoryScore"] = match["score"] if match else None
                    update_fields["needsReview"] = match is None

                operations.append(UpdateOne({"_id": doc["_id"]}, {"$set": update_fields}))

            result = collection.bulk_write(operations)
            print(f"[backfill] embedded {result.modified_count} docs "
                  f"({i + len(chunk)}/{len(docs)} processed)"
                  + (" + matched to categories" if run_matching else ""))
    else:
        print("[backfill] nothing to do")

    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if vector_index_name in existing:
        print(f"[backfill] index '{vector_index_name}' already exists, skipping")
    else:
        collection.create_search_index({
            "name": vector_index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": num_dimensions, "similarity": "cosine"}
                ]
            },
        })
        print(f"[backfill] created index '{vector_index_name}' ({num_dimensions} dims). "
              f"Atlas needs ~1-2 min to finish building it.")

    print(f"[backfill] DONE -- '{collection_name}' is now searchable via '{vector_index_name}'")