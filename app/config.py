import os
from dotenv import load_dotenv

load_dotenv()

# Groq — text generation (menu extraction)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # good accuracy/speed balance; swap models as needed

# Ollama — embeddings (keep this, Groq has no embeddings endpoint)
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "bizlink"

# Spring Boot backend
CATEGORIES_API_URL = "https://dizziness-pasted-scarecrow.ngrok-free.dev/api/product/categories"

# Upload constraints
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp",
    "application/pdf",
    "text/plain",
    "text/csv", "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}