from app.database import db

bad_docs = list(db.products.find({"normalizedName": {"$regex": "```"}}))

print(f"Found {len(bad_docs)} product(s) with corrupted normalizedName:\n")
for doc in bad_docs:
    print(f"_id: {doc['_id']}")
    print(f"name: {doc['name']}")
    print(f"normalizedName (first 100 chars): {doc['normalizedName'][:100]}...")
    print("-" * 40)