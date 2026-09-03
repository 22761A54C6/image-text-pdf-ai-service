import os
import tempfile
import json
import logging
from typing import List
import uuid
from datetime import datetime
import asyncio
import aiohttp
from functools import wraps
import time

import uvicorn
from app.config import HOST, PORT
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse

from app.document_loader import load_document_text, load_pdf_text
from app.config import ALLOWED_CONTENT_TYPES, ALLOWED_PDF_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.models import ExtractResponse, Product, TextExtractRequest
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_with_llm
from app.embeddings import embed_and_store
from app.database import db, catalog_db

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Configuration from environment variables
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
OPENSEARCH_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "5"))
OPENSEARCH_RETRIES = int(os.getenv("OPENSEARCH_RETRIES", "3"))
OPENSEARCH_RETRY_DELAY = int(os.getenv("OPENSEARCH_RETRY_DELAY", "1"))

# Circuit breaker state
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        return self.state

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
        return self.state

    def can_attempt(self):
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open allows one attempt

circuit_breaker = CircuitBreaker()

# Async HTTP session for connection pooling
async def get_aiohttp_session():
    if not hasattr(get_aiohttp_session, "session") or get_aiohttp_session.session.closed:
        timeout = aiohttp.ClientTimeout(total=OPENSEARCH_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        get_aiohttp_session.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return get_aiohttp_session.session

async def close_aiohttp_session():
    if hasattr(get_aiohttp_session, "session") and not get_aiohttp_session.session.closed:
        await get_aiohttp_session.session.close()

async def send_to_opensearch(log_data):
    """Send log data directly to OpenSearch via HTTP with retry logic and circuit breaker"""
    if not circuit_breaker.can_attempt():
        return  # Circuit breaker is open, skip

    index_name = f"image-text-pdf-logs-{datetime.utcnow().strftime('%Y.%m.%d')}"
    url = f"{OPENSEARCH_HOST}/{index_name}/_doc"

    for attempt in range(OPENSEARCH_RETRIES):
        try:
            session = await get_aiohttp_session()
            async with session.post(url, json=log_data) as response:
                if response.status < 400:
                    circuit_breaker.record_success()
                    return
                elif response.status >= 500:
                    # Server error, retry
                    await asyncio.sleep(OPENSEARCH_RETRY_DELAY * (2 ** attempt))
                else:
                    # Client error, don't retry
                    circuit_breaker.record_failure()
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < OPENSEARCH_RETRIES - 1:
                await asyncio.sleep(OPENSEARCH_RETRY_DELAY * (2 ** attempt))
            else:
                circuit_breaker.record_failure()
                # Log the error silently to avoid spamming
                pass

app = FastAPI(title="Menu AI Extraction Service")

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await send_to_opensearch({"event": "shutdown", "message": "Service shutting down", "@timestamp": datetime.utcnow().isoformat()})
    await close_aiohttp_session()

@app.on_event("startup")
async def startup_sync_categories():
    from app.sync_categories import sync_categories
    print("[startup] Syncing categories from catalog MongoDB...")
    try:
        sync_categories()
        print("[startup] Category sync completed successfully")
    except Exception as e:
        print(f"[startup] Category sync failed: {e}")
        # Don't crash the whole service if sync fails

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with status codes"""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()

    log_data = {
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration * 1000, 2),
        "@timestamp": datetime.utcnow().isoformat()
    }
    await send_to_opensearch(log_data)
    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Safety net for anything not already raised as an HTTPException --
    ensures the client always gets clean JSON instead of a raw 500/traceback."""
    await send_to_opensearch({
        "event": "unhandled_exception",
        "method": request.method,
        "path": request.url.path,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "@timestamp": datetime.utcnow().isoformat()
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/api/top-categories")
def get_categories():
    """
    Reads top-level categories directly from Spring Boot's own Mongo
    database (catalog.categories on 192.168.0.109:27017) -- same data
    you'd get from GET http://192.168.0.109:8081/api/product/categories,
    but read straight from Mongo instead of an HTTP call to Spring Boot.
    Subcategories (docs with a non-null parentCategoryId) are excluded,
    matching the top-level-only filtering used in sync_categories.py.
    Note: these docs also have an unrelated "parentId" field -- that is
    not the category hierarchy link, so it's ignored here.
    """
    docs = catalog_db.categories.find({"parentCategoryId": {"$exists": False}})

    categories = []
    for doc in docs:
        categories.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
        })

    return categories


@app.post("/sync/categories")
def sync_categories_endpoint():
    """
    Manually trigger category sync from catalog MongoDB to bizlink MongoDB.
    Call this endpoint when frontend adds new categories to refresh the
    vector index with the latest categories.
    """
    from app.sync_categories import sync_categories
    try:
        sync_categories()
        return {"status": "success", "message": "Categories synced successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Category sync failed: {e}")


@app.post("/image")
async def extract(file: UploadFile = File(...)):
    await send_to_opensearch({"event": "extract", "message": "Endpoint hit", "@timestamp": datetime.utcnow().isoformat()})

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
        )

    contents = await file.read()
    file_size = len(contents)
    await send_to_opensearch({"event": "extract", "message": "File received", "filename": file.filename, "file_size_bytes": file_size, "@timestamp": datetime.utcnow().isoformat()})
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        raw_text = load_document_text(tmp_path, file.content_type)
        await send_to_opensearch({"event": "extract", "message": "OCR completed", "text_length": len(raw_text) if raw_text else 0, "@timestamp": datetime.utcnow().isoformat()})

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in image")

        products = extract_products_with_llm(raw_text)
        await send_to_opensearch({"event": "extract", "message": "LLM extraction completed", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

        normalized_products = []
        for p in products:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            price = p.get("price")
            if not isinstance(price, (int, float)):
                price = 0.0
            normalized_products.append({"name": p["name"], "price": float(price)})

        products = normalized_products
        await send_to_opensearch({"event": "extract", "message": "Products normalized", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

        if not products:
            raise HTTPException(status_code=422, detail="No products could be extracted from this image")

        batch_id = str(uuid.uuid4())

        await send_to_opensearch({"event": "extract", "message": "Sending to embed_and_store", "batch_id": batch_id, "products_count": len(products), "@timestamp": datetime.utcnow().isoformat()})
        try:
            stored_docs = embed_and_store(products, batch_id=batch_id)
            await send_to_opensearch({"event": "extract", "message": "embed_and_store completed", "stored_count": len(stored_docs), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
        except Exception as e:
            await send_to_opensearch({"event": "extract", "message": "embed_and_store failed", "error": str(e), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
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
    await send_to_opensearch({"event": "pdf", "message": "Endpoint hit", "@timestamp": datetime.utcnow().isoformat()})

    if file.content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only application/pdf is allowed."
        )

    contents = await file.read()
    file_size = len(contents)
    await send_to_opensearch({"event": "pdf", "message": "File received", "filename": file.filename, "file_size_bytes": file_size, "@timestamp": datetime.utcnow().isoformat()})
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        raw_text = load_pdf_text(tmp_path)
        await send_to_opensearch({"event": "pdf", "message": "PDF text extraction completed", "text_length": len(raw_text) if raw_text else 0, "@timestamp": datetime.utcnow().isoformat()})

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in PDF")

        products = extract_products_with_llm(raw_text)
        await send_to_opensearch({"event": "pdf", "message": "LLM extraction completed", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

        normalized_products = []
        for p in products:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            price = p.get("price")
            if not isinstance(price, (int, float)):
                price = 0.0
            normalized_products.append({"name": p["name"], "price": float(price)})

        products = normalized_products
        await send_to_opensearch({"event": "pdf", "message": "Products normalized", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

        if not products:
            raise HTTPException(status_code=422, detail="No products could be extracted from this PDF")

        batch_id = str(uuid.uuid4())

        await send_to_opensearch({"event": "pdf", "message": "Sending to embed_and_store", "batch_id": batch_id, "products_count": len(products), "@timestamp": datetime.utcnow().isoformat()})
        try:
            stored_docs = embed_and_store(products, batch_id=batch_id)
            await send_to_opensearch({"event": "pdf", "message": "embed_and_store completed", "stored_count": len(stored_docs), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
        except Exception as e:
            await send_to_opensearch({"event": "pdf", "message": "embed_and_store failed", "error": str(e), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
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
async def get_text(payload: TextExtractRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    await send_to_opensearch({"event": "getText", "message": "Endpoint hit", "text_length": len(payload.text), "@timestamp": datetime.utcnow().isoformat()})

    products = extract_products_with_llm(payload.text)
    await send_to_opensearch({"event": "getText", "message": "LLM extraction completed", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

    normalized_products = []
    for p in products:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        price = p.get("price")
        if not isinstance(price, (int, float)):
            price = 0.0
        normalized_products.append({"name": p["name"], "price": float(price)})

    products = normalized_products
    await send_to_opensearch({"event": "getText", "message": "Products normalized", "products_count": len(products), "products": products, "@timestamp": datetime.utcnow().isoformat()})

    if not products:
        raise HTTPException(status_code=422, detail="No products could be extracted from this text")

    batch_id = str(uuid.uuid4())

    await send_to_opensearch({"event": "getText", "message": "Sending to embed_and_store", "batch_id": batch_id, "products_count": len(products), "@timestamp": datetime.utcnow().isoformat()})
    try:
        stored_docs = embed_and_store(products, batch_id=batch_id)
        await send_to_opensearch({"event": "getText", "message": "embed_and_store completed", "stored_count": len(stored_docs), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        await send_to_opensearch({"event": "getText", "message": "embed_and_store failed", "error": str(e), "batch_id": batch_id, "@timestamp": datetime.utcnow().isoformat()})
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6000)