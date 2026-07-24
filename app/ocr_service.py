import cv2
import logging
from fastapi import HTTPException
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.WARNING)

# Loaded once at import time (heavy — don't reload per request)
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")


def preprocess_image(image_path: str):
    """Upscale only if small. Denoise + Otsu threshold were removed —
    testing showed they caused missed lines (e.g. Tempura Haddock,
    Herb-Crusted Monkfish vanished) and character-level misreads
    (e.g. 'Style' -> 'Styte', 'Pollock' -> 'Pallock')."""
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    target_width = 2400  # more aggressive than before
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return img


def run_ocr(processed_img) -> str:
    try:
        result = ocr_engine.ocr(processed_img, cls=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed to process image: {e}")

    lines = []
    for block in result or []:
        if not block:
            continue
        for line in block:
            try:
                text = line[1][0]
                lines.append(text)
            except (IndexError, TypeError):
                continue
    return "\n".join(lines)


def run_ocr_on_array(img_array) -> str:
    """Same OCR logic as run_ocr, but for an in-memory image array
    (used for PDF pages rendered directly from PyMuPDF, no file path)."""
    try:
        result = ocr_engine.ocr(img_array, cls=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed to process PDF page: {e}")

    lines = []
    for block in result or []:
        if not block:
            continue
        for line in block:
            try:
                text = line[1][0]
                lines.append(text)
            except (IndexError, TypeError):
                continue
    return "\n".join(lines)