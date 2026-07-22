from fastapi import HTTPException

from app.database import db

AUTO_MAP_THRESHOLD = 0.95
VENDOR_CONFIRM_THRESHOLD = 0.85


def find_nearest_category(product_embedding: list[float], top_k: int = 1) -> list[dict]:
    """Runs MongoDB Atlas Vector Search to find the closest category
    to a given product embedding. Returns matches with similarity scores.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "category_vector_index",
                "path": "embedding",
                "queryVector": product_embedding,
                "numCandidates": 512,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "sourceId": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    try:
        return list(db.categories.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Category vector search failed: {e}")


def classify_match(product_embedding: list[float]) -> dict:
    """
    Tiered category matching:
      score > 0.95         -> AUTO_MAPPED
      0.85 <= score <= 0.95 -> PENDING_VENDOR_CONFIRMATION
      score < 0.85 or no match -> CREATE_NEW_CATEGORY_OR_ADMIN_APPROVAL
    """
    matches = find_nearest_category(product_embedding, top_k=1)

    if not matches:
        return {
            "status": "CREATE_NEW_CATEGORY_OR_ADMIN_APPROVAL",
            "matchedCategory": None,
            "matchedCategorySourceId": None,
            "matchedCategoryScore": None,
        }

    top = matches[0]
    score = top["score"]

    if score > AUTO_MAP_THRESHOLD:
        status = "AUTO_MAPPED"
    elif score >= VENDOR_CONFIRM_THRESHOLD:
        status = "PENDING_VENDOR_CONFIRMATION"
    else:
        status = "CREATE_NEW_CATEGORY_OR_ADMIN_APPROVAL"

    return {
        "status": status,
        "matchedCategory": top["name"],
        "matchedCategorySourceId": top["sourceId"],
        "matchedCategoryScore": score,
    }