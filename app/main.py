import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.models import ExtractResponse
from app.ocr_service import preprocess_image, run_ocr
from app.extraction import extract_products_regex, extract_products_with_llm
from app.embeddings import embed_and_store
from app.sync_categories import sync_categories

app = FastAPI(title="Menu AI Extraction Service")


@app.on_event("startup")
def startup_sync_categories():
    print("[startup] Syncing categories from Spring Boot API...")
    sync_categories()


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

        processed = preprocess_image(tmp_path)
        raw_text = run_ocr(processed)

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text detected in image")

        products = extract_products_regex(raw_text)

        if not products:
            products = extract_products_with_llm(raw_text)

        embed_and_store(products)

        return ExtractResponse(products=products)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}