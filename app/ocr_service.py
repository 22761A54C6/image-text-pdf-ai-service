import cv2
from fastapi import HTTPException
from paddleocr import PaddleOCR

# Loaded once at import time (heavy — don't reload per request)
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")


def preprocess_image(image_path: str):
    """Grayscale -> denoise -> threshold -> upscale if small.
    Improves OCR accuracy on photographed (non-scanned) menus.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")
    #Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Upscale if small (improves OCR accuracy)
    h, w = thresh.shape
    if w < 1200:
        scale = 1200 / w
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return thresh


def run_ocr(processed_img) -> str:
    result = ocr_engine.ocr(processed_img, cls=True)
    lines = []
    for block in result:
        if not block:
            continue
        for line in block:
            text = line[1][0]
            lines.append(text)
    return "\n".join(lines)
