import os
from dotenv import load_dotenv

load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL = "llama-3.3-70b-versatile"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_DIMENSIONS = 512
GEMINI_TEXT_MODEL="gemini-3.5-flash"
# GEMINI_TEXT_MODEL = "gemini-2.5-flash"
# GEMINI_TEXT_MODEL = "gemini-2.5-flash-lite"

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "bizlink"

CATEGORIES_API_URL = "https://dizziness-pasted-scarecrow.ngrok-free.dev/api/product/categories"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024