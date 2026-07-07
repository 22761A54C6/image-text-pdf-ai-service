from fastapi import HTTPException

from app.ocr_service import preprocess_image, run_ocr


IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}


def load_document_text(file_path: str, content_type: str) -> str:
    if content_type in IMAGE_TYPES:
        processed = preprocess_image(file_path)
        return run_ocr(processed)
    raise HTTPException(status_code=400, detail=f"Unsupported document type: {content_type}")