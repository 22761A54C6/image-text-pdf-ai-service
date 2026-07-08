import os
import tempfile
import json
from typing import List
import uuid

import uvicorn
from app.config import HOST, PORT
from fastapi import FastAPI, HTTPException, UploadFile, File

from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.document_loader import load_document_text
from app.models import ExtractResponse, Product
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_with_llm
from app.embeddings import embed_and_store
from app.sync_categories import sync_categories
from app.database import db

app = FastAPI(title="Menu AI Extraction Service")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)


@app.on_event("startup")
def startup_sync_categories():
    print("[startup] Syncing categories from Spring Boot API...")
    sync_categories()
import asyncio

@app.post("/extract")
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

        batch_id = str(uuid.uuid4())

        # Actually persist to MongoDB (this is what was missing)
        print(f"[extract] >>> SENDING {len(products)} PRODUCTS TO embed_and_store/MongoDB <<< batchId={batch_id}")
        try:
            stored_docs = embed_and_store(products, batch_id=batch_id)
            print(f"[extract] embed_and_store stored {len(stored_docs)} of {len(products)} products")
        except Exception as e:
            print(f"[extract] !!! embed_and_store FAILED: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store products: {e}")

        # Attach batchId to each product for plain JSON response
        for p in products:
            p["batchId"] = batch_id

        return {
            "batchId": batch_id,
            "products": products
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/products", response_model=List[Product])
def get_products():
    docs = db.products.find({}, {"embedding": 0})  # exclude embedding, it's huge and not useful here

    products = []
    for doc in docs:
        products.append({
            "id": doc["_id"],
            "batchId": doc.get("batchId"),
            "name": doc.get("name"),
            "normalizedName": doc.get("normalizedName"),
            "price": doc.get("price"),
            "matchedCategory": doc.get("matchedCategory"),
            "matchedCategorySourceId": doc.get("matchedCategorySourceId"),
            "matchedCategoryScore": doc.get("matchedCategoryScore"),
            "matchStatus": doc.get("matchStatus"),
        })

    return products


@app.get("/product/{source_id}", response_model=List[Product])
def get_products_by_category(source_id: str):
    docs = db.products.find({"matchedCategorySourceId": source_id}, {"embedding": 0})

    products = []
    for doc in docs:
        products.append({
            "id": doc["_id"],
            "batchId": doc.get("batchId"),
            "name": doc.get("name"),
            "normalizedName": doc.get("normalizedName"),
            "price": doc.get("price"),
            "matchedCategory": doc.get("matchedCategory"),
            "matchedCategorySourceId": doc.get("matchedCategorySourceId"),
            "matchedCategoryScore": doc.get("matchedCategoryScore"),
            "matchStatus": doc.get("matchStatus"),
        })

    if not products:
        raise HTTPException(status_code=404, detail=f"No products found for category sourceId '{source_id}'")

    return products


@app.get("/products/batch/{batch_id}", response_model=List[Product])
def get_products_by_batch(batch_id: str):
    docs = db.products.find({"batchId": batch_id}, {"embedding": 0})

    products = []
    for doc in docs:
        products.append({
            "id": doc["_id"],
            "batchId": doc.get("batchId"),
            "name": doc.get("name"),
            "normalizedName": doc.get("normalizedName"),
            "price": doc.get("price"),
            "matchedCategory": doc.get("matchedCategory"),
            "matchedCategorySourceId": doc.get("matchedCategorySourceId"),
            "matchedCategoryScore": doc.get("matchedCategoryScore"),
            "matchStatus": doc.get("matchStatus"),
        })

    if not products:
        raise HTTPException(status_code=404, detail=f"No products found for batchId '{batch_id}'")

    return products


@app.get("/health")
def health():
    return {"status": "ok"}