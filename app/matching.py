from app.database import db

# Below this score, don't trust the match — better to leave it
# unmatched than confidently assign the wrong category.
MIN_CONFIDENCE_SCORE = 0.80


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
                "score": {"$meta": "vectorSearchScore"}   #categories are stored in the db with a field called "embedding"
            }
        }
    ]

    return list(db.categories.aggregate(pipeline))



def find_confident_category(product_embedding: list[float]) -> dict | None:
    matches = find_nearest_category(product_embedding, top_k=1)
    if not matches:
        return None

    top_match = matches[0]
    # Temporarily disabled to observe real scores:
    if top_match["score"] < MIN_CONFIDENCE_SCORE:
        return None

    return top_match