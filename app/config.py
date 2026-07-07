import os
from dotenv import load_dotenv

load_dotenv()

# Groq — text generation (menu extraction)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # good accuracy/speed balance; swap models as needed


# Voyage AI — embeddings (Groq has no embeddings endpoint, Voyage is MongoDB's
# own embedding provider so it pairs naturally with Atlas Vector Search)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_EMBED_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_EMBED_MODEL = "voyage-4-lite"   # current gen; replaces voyage-3.5-lite
VOYAGE_EMBED_DIMENSIONS = 512          # model default is 1024; we request 512 explicitly to keep storage small

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "bizlink"

# Spring Boot backend
CATEGORIES_API_URL = "https://dizziness-pasted-scarecrow.ngrok-free.dev/api/product/categories"

# Upload constraints -- images only (PDF/CSV/DOCX support removed)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB