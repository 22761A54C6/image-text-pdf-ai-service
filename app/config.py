import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.environ.get("PORT", 6000))
HOST = os.environ.get("HOST", "127.0.0.1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_DIMENSIONS = 512
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "bizlink"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
ALLOWED_PDF_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024