import os
from dotenv import load_dotenv

load_dotenv()

# Ollama — text generation (menu extraction fallback)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"  # pulled via: `ollama pull qwen2.5:3b`

# Ollama — embeddings
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # pulled via: `ollama pull nomic-embed-text`

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "bizlink"

# Spring Boot backend
CATEGORIES_API_URL = "https://dizziness-pasted-scarecrow.ngrok-free.dev/api/product/categories"

# Upload constraints
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
