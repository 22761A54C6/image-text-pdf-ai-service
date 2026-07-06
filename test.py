# import sys
# sys.path.insert(0, ".")

# from app.ocr_service import preprocess_image, run_ocr
# from app.extraction import extract_products_regex, extract_products_with_llm

# image_path = "C:\\Users\\user\\Downloads\\image_PI005.jpg"

# processed = preprocess_image(image_path)
# raw_text = run_ocr(processed)

# print("===== RAW OCR OUTPUT (WITH OpenCV preprocessing) =====")
# print(raw_text)
# print("=" * 50)
# print(f"Lines detected: {len([l for l in raw_text.split(chr(10)) if l.strip()])}")

# print("\n===== REGEX EXTRACTION =====")
# regex_products = extract_products_regex(raw_text)
# print(f"Products found: {len(regex_products)}")
# for p in regex_products:
#     print(f"  {p}")

# print("\n===== LLM EXTRACTION (Ollama qwen2.5:3b) =====")
# llm_products = extract_products_with_llm(raw_text)
# print(f"Products found: {len(llm_products)}")
# for p in llm_products:
#     print(f"  {p}")


import sys
sys.path.insert(0, ".")
from app.ocr_service import preprocess_image, run_ocr

image_path = r"C:\Users\user\Downloads\image_PI009.jpg"
processed = preprocess_image(image_path)
raw_text = run_ocr(processed)
print(raw_text)
print(f"Lines: {len(raw_text.splitlines())}")