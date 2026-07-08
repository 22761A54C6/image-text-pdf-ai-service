# # # # import sys
# # # # sys.path.insert(0, ".")

# # # # from app.ocr_service import preprocess_image, run_ocr
# # # # from app.extraction import extract_products_regex, extract_products_with_llm

# # # # image_path = "C:\\Users\\user\\Downloads\\image_PI005.jpg"

# # # # processed = preprocess_image(image_path)
# # # # raw_text = run_ocr(processed)

# # # # print("===== RAW OCR OUTPUT (WITH OpenCV preprocessing) =====")
# # # # print(raw_text)
# # # # print("=" * 50)
# # # # print(f"Lines detected: {len([l for l in raw_text.split(chr(10)) if l.strip()])}")

# # # # print("\n===== REGEX EXTRACTION =====")
# # # # regex_products = extract_products_regex(raw_text)
# # # # print(f"Products found: {len(regex_products)}")
# # # # for p in regex_products:
# # # #     print(f"  {p}")

# # # # print("\n===== LLM EXTRACTION (Ollama qwen2.5:3b) =====")
# # # # llm_products = extract_products_with_llm(raw_text)
# # # # print(f"Products found: {len(llm_products)}")
# # # # for p in llm_products:
# # # #     print(f"  {p}")


# # # import sys
# # # sys.path.insert(0, ".")
# # # from app.ocr_service import preprocess_image, run_ocr

# # # image_path = r"C:\Users\user\Downloads\image_PI009.jpg"
# # # processed = preprocess_image(image_path)
# # # raw_text = run_ocr(processed)
# # # print(raw_text)
# # # print(f"Lines: {len(raw_text.splitlines())}")



# # from app.database import db

# # # 1. Confirm it's really 768
# # for idx in db["categories"].list_search_indexes():
# #     print(idx)

# # # 2. Drop it
# # db["categories"].drop_search_index("category_vector_index")
# # print("drop command sent")

# # # 3. Wait, then confirm it's gone
# # import time
# # time.sleep(45)
# # for idx in db["categories"].list_search_indexes():
# #     print(idx)


# from app.database import db

# indexes = list(db["categories"].list_search_indexes())
# if not indexes:
#     print("No search indexes exist on 'categories' -- clean, ready to be recreated at 512 dims.")
# else:
#     for idx in indexes:
#         print(idx)

# from app.database import db

# result_c = db.categories.update_many({}, {"$unset": {"embedding": ""}})
# result_p = db.products.update_many({}, {"$unset": {"embedding": ""}})

# print(f"Cleared embeddings on {result_c.modified_count} categories")
# print(f"Cleared embeddings on {result_p.modified_count} products")

from app.database import db

bad_docs = list(db.products.find({"normalizedName": {"$regex": "```"}}))

print(f"Found {len(bad_docs)} product(s) with corrupted normalizedName:\n")
for doc in bad_docs:
    print(f"_id: {doc['_id']}")
    print(f"name: {doc['name']}")
    print(f"normalizedName (first 100 chars): {doc['normalizedName'][:100]}...")
    print("-" * 40)