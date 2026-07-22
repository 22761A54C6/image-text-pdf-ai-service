import uuid
from typing import List

from google import genai
from google.genai.types import EmbedContentConfig
from fastapi import HTTPException

from app.config import GEMINI_API_KEY, GEMINI_EMBED_MODEL, GEMINI_EMBED_DIMENSIONS
from app.database import db
from app.matching import classify_match
from app.normalization import normalize_product_names_batch

_client = genai.Client()  # reads GEMINI_API_KEY from env automatically
_TASK_TYPE = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


def get_embedding(text: str, input_type: str = "document") -> List[float]:
    """input_type: "document" for things you store, "query" for search text.
    Gemini tunes embeddings differently per task_type -- getting this wrong
    doesn't error, it just quietly makes matches worse."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set -- add it to your .env file")

    try:
        result = _client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=[text],
            config=EmbedContentConfig(
                task_type=_TASK_TYPE.get(input_type, "RETRIEVAL_DOCUMENT"),
                output_dimensionality=GEMINI_EMBED_DIMENSIONS,
            ),
        )
        return result.embeddings[0].values
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini embedding request failed: {e}")


def get_embeddings_batch(texts: List[str], input_type: str = "document") -> List[List[float]]:
    """Batch version for the injection pipeline."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set -- add it to your .env file")

    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        try:
            result = _client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=chunk,
                config=EmbedContentConfig(
                    task_type=_TASK_TYPE.get(input_type, "RETRIEVAL_DOCUMENT"),
                    output_dimensionality=GEMINI_EMBED_DIMENSIONS,
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini embedding request failed on batch {i // batch_size + 1}: {e}"
            )
        embeddings.extend(e.values for e in result.embeddings)
    return embeddings


def embed_and_store(products: List[dict], batch_id: str = None) -> List[dict]:
    if not products:
        print("[embed_and_store] no products to store")
        return []

    raw_names = [p["name"] for p in products]
    normalized_names = normalize_product_names_batch(raw_names)
    print(f"[embed_and_store] normalized {len(normalized_names)} names via Gemini")

    # Let embedding failures propagate -- silently returning [] here previously
    # made callers believe storage succeeded with 0 products saved.
    embeddings = get_embeddings_batch(normalized_names, input_type="document")

    print(f"[embed_and_store] {len(normalized_names)} names in, {len(embeddings)} embeddings out")
    if len(embeddings) != len(normalized_names):
        raise HTTPException(
            status_code=502,
            detail=f"Embedding count mismatch: got {len(embeddings)} embeddings for "
                   f"{len(normalized_names)} products"
        )

    docs = []
    for product, normalized_name, embedding in zip(products, normalized_names, embeddings):
        try:
            classification = classify_match(embedding)
            docs.append({
                "_id": str(uuid.uuid4()),
                "batchId": batch_id,
                "name": product["name"],
                "normalizedName": normalized_name,
                "price": product.get("price"),
                "embedding": embedding,
                "matchedCategory": classification["matchedCategory"],
                "matchedCategorySourceId": classification["matchedCategorySourceId"],
                "matchedCategoryScore": classification["matchedCategoryScore"],
                "matchStatus": classification["status"],
            })
        except Exception as e:
            print(f"[embed_and_store] failed to process '{product.get('name')}': {e}")

    print(f"[embed_and_store] {len(products)} extracted, {len(docs)} survived to insert")

    if not docs:
        print("[embed_and_store] all products failed to process, nothing inserted")
        return []

    result = db.products.insert_many(docs)
    print(f"[embed_and_store] inserted {len(result.inserted_ids)} of {len(products)} products")
    return docs