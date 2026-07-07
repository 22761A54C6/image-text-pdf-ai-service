import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.document_loader import load_document_text
from app.models import ExtractResponse
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_with_llm
from app.embeddings import embed_and_store, backfill_embeddings
from app.sync_categories import sync_categories

app = FastAPI(title="Menu AI Extraction Service")


@app.on_event("startup")
def startup_sync_categories():
    print("[startup] Syncing categories from Spring Boot API...")
    sync_categories()

    print("[startup] Backfilling embeddings for any un-embedded products...")
    try:
        backfill_embeddings(
            collection_name="products",
            text_field="name",
            vector_index_name="product_vector_index",
            run_matching=True,
        )
    except Exception as e:
        print(f"[startup] Products backfill failed, server will start anyway: {e}")


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
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

        # processed = preprocess_image(tmp_path)
        # raw_text = run_ocr(processed)

        raw_text = load_document_text(tmp_path, file.content_type)

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in image")

        products = extract_products_with_llm(raw_text)

        # Normalize: keep every product even if price is missing/invalid,
        # default price to 0.0 rather than dropping the item. Only skip
        # items with no usable name at all.
        normalized_products = []
        for p in products:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            price = p.get("price")
            if not isinstance(price, (int, float)):
                price = 0.0
            normalized_products.append({"name": p["name"], "price": float(price)})

        products = normalized_products

        if not products:
            raise HTTPException(status_code=422, detail="No products could be extracted from this image")

        embed_and_store(products)

        return ExtractResponse(products=products)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}