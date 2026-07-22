import os
import tempfile
import json
import logging
from typing import List
import uuid

import uvicorn
from app.config import HOST, PORT
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse

from app.document_loader import load_document_text, load_pdf_text
from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.models import ExtractResponse, Product, TextExtractRequest
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_with_llm
from app.embeddings import embed_and_store
from app.sync_categories import sync_categories
from app.database import db

ALLOWED_PDF_CONTENT_TYPES = {"application/pdf"}

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Menu AI Extraction Service")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Safety net for anything not already raised as an HTTPException --
    ensures the client always gets clean JSON instead of a raw 500/traceback."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)


@app.on_event("startup")
def startup_sync_categories():
    print("[startup] Syncing categories from Spring Boot API...")
    try:
        sync_categories()
    except Exception as e:
        # Don't crash the whole service if the Spring Boot API is unreachable
        # at boot time -- log it and let /sync/categories be retried later.
        print(f"[startup] Category sync failed, continuing without fresh categories: {e}")


@app.post("/sync/categories")
def trigger_category_sync():
    """
    Re-fetch categories from the Spring Boot API and reconcile Mongo:
    adds new categories, updates changed ones, deletes ones no longer
    present upstream.
    """
    try:
        sync_categories()
        return {"status": "ok", "message": "Category sync completed"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Category sync failed: {e}")


@app.post("/image")
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

        print(f"[extract] >>> SENDING {len(products)} PRODUCTS TO embed_and_store/MongoDB <<< batchId={batch_id}")
        try:
            stored_docs = embed_and_store(products, batch_id=batch_id)
            print(f"[extract] embed_and_store stored {len(stored_docs)} of {len(products)} products")
        except Exception as e:
            print(f"[extract] !!! embed_and_store FAILED: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store products: {e}")

        for p in products:
            p["batchId"] = batch_id

        return {
            "batchId": batch_id,
            "products": products
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/products/image/{batch_id}", response_model=List[Product])
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


@app.post("/pdf")
async def extract_pdf(file: UploadFile = File(...)):
    print("!!!!!!!!!! PDF EXTRACT ENDPOINT HIT !!!!!!!!!!")

    if file.content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only application/pdf is allowed."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        raw_text = load_pdf_text(tmp_path)
        print(f"[pdf] raw_text length={len(raw_text) if raw_text else 0}")

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in PDF")

        products = extract_products_with_llm(raw_text)
        print(f"[pdf] LLM extracted {len(products)} raw products: "
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
        print(f"[pdf] {len(products)} products after normalization: "
              f"{json.dumps(products, ensure_ascii=False)}")

        if not products:
            raise HTTPException(status_code=422, detail="No products could be extracted from this PDF")

        batch_id = str(uuid.uuid4())

        print(f"[pdf] >>> SENDING {len(products)} PRODUCTS TO embed_and_store/MongoDB <<< batchId={batch_id}")
        try:
            stored_docs = embed_and_store(products, batch_id=batch_id)
            print(f"[pdf] embed_and_store stored {len(stored_docs)} of {len(products)} products")
        except Exception as e:
            print(f"[pdf] !!! embed_and_store FAILED: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store products: {e}")

        for p in products:
            p["batchId"] = batch_id

        return {
            "batchId": batch_id,
            "products": products
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/products/pdf/{batch_id}", response_model=List[Product])
def get_pdf_products_by_batch(batch_id: str):
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


@app.post("/getText")
def get_text(payload: TextExtractRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    print(f"[getText] raw_text length={len(payload.text)}")

    products = extract_products_with_llm(payload.text)
    print(f"[getText] LLM extracted {len(products)} raw products: "
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
    print(f"[getText] {len(products)} products after normalization: "
          f"{json.dumps(products, ensure_ascii=False)}")

    if not products:
        raise HTTPException(status_code=422, detail="No products could be extracted from this text")

    batch_id = str(uuid.uuid4())

    print(f"[getText] >>> SENDING {len(products)} PRODUCTS TO embed_and_store/MongoDB <<< batchId={batch_id}")
    try:
        stored_docs = embed_and_store(products, batch_id=batch_id)
        print(f"[getText] embed_and_store stored {len(stored_docs)} of {len(products)} products")
    except Exception as e:
        print(f"[getText] !!! embed_and_store FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store products: {e}")

    for p in products:
        p["batchId"] = batch_id

    return {
        "batchId": batch_id,
        "products": products
    }


@app.get("/products/text/{batch_id}", response_model=List[Product])
def get_text_batch(batch_id: str):
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