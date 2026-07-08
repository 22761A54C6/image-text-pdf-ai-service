import os
import tempfile
import json

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.document_loader import load_document_text
from app.models import ExtractResponse
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_with_llm
from app.embeddings import embed_and_store
from app.sync_categories import sync_categories

app = FastAPI(title="Menu AI Extraction Service")


@app.on_event("startup")
def startup_sync_categories():
    print("[startup] Syncing categories from Spring Boot API...")
    sync_categories()


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    print("!!!!!!!!!! EXTRACT ENDPOINT HIT !!!!!!!!!!")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        raw_text = load_document_text(tmp_path, file.content_type)
        print(f"[extract] OCR raw_text length={len(raw_text) if raw_text else 0}")

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in image")

        products = extract_products_with_llm(raw_text)
        print(f"[extract] LLM extracted {len(products)} raw products: "
              f"{json.dumps(products, ensure_ascii=False)}")

        normalized_products = []
        for p in products:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            price = p.get("price")
            if not isinstance(price, (int, float)):
                price = 0.0
            normalized_products.append({"name": p["name"], "price": float(price)})

        products = normalized_products
        print(f"[extract] {len(products)} products after normalization: "
              f"{json.dumps(products, ensure_ascii=False)}")

        if not products:
            raise HTTPException(status_code=422, detail="No products could be extracted from this image")

        print(f"[extract] >>> SENDING {len(products)} PRODUCTS TO embed_and_store/MongoDB <<<")
        embed_and_store(products)

        return ExtractResponse(products=products)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}