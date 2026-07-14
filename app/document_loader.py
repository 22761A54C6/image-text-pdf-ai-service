import fitz  # PyMuPDF
import numpy as np
from fastapi import HTTPException

from app.ocr_service import preprocess_image, run_ocr, run_ocr_on_array


IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
MIN_TEXT_LAYER_CHARS = 20  # below this, treat page as scanned/no real text layer


def load_document_text(file_path: str, content_type: str) -> str:
    if content_type in IMAGE_TYPES:
        processed = preprocess_image(file_path)
        return run_ocr(processed)
    raise HTTPException(status_code=400, detail=f"Unsupported document type: {content_type}")


def load_pdf_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    all_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) >= MIN_TEXT_LAYER_CHARS:
            print(f"[pdf] page {page_num + 1}: found text layer ({len(text)} chars), skipping OCR")
            all_text.append(text)
        else:
            print(f"[pdf] page {page_num + 1}: no usable text layer ({len(text)} chars), falling back to OCR")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img_array = img_array[:, :, [2, 1, 0]]
            elif pix.n == 3:
                img_array = img_array[:, :, ::-1]

            ocr_text = run_ocr_on_array(img_array)
            all_text.append(ocr_text)

    page_count = len(doc)
    doc.close()

    combined = "\n".join(t for t in all_text if t)
    print(f"[pdf] extracted {len(combined)} total chars from {page_count} page(s)")
    return combined