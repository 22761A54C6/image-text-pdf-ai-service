# Image-Text-Pdf AI Service

A FastAPI-based service for extracting product information from images, PDFs, and raw text using OCR and LLM processing.

## Features

- **Image OCR**: Extract text from menu images using PaddleOCR
- **PDF Processing**: Extract text from PDFs with OCR fallback for scanned documents
- **Text Extraction**: Process raw text input
- **LLM Integration**: Use Gemini AI for intelligent product extraction
- **Product Matching**: Vector-based category matching using MongoDB Atlas
- **Logging**: Centralized logging to OpenSearch with circuit breaker pattern

## Prerequisites

- Python 3.11.10 or higher (recommended -Python 3.11.10 is the latest stable version)
- MongoDB (with Atlas Vector Search enabled)
- OpenSearch (for logging)
- Gemini API Key(Present using Free Google AI Studio for testing)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Image-Text-Pdf-AI-Service
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Service Configuration
   PORT=6000
   HOST=127.0.0.1

   # Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here

   # MongoDB
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/bizlink
   MONGO_DB_NAME=bizlink

   # OpenSearch (Logging)
   OPENSEARCH_HOST=http://localhost:9200
   OPENSEARCH_TIMEOUT=5
   OPENSEARCH_RETRIES=3
   OPENSEARCH_RETRY_DELAY=1

   # Categories API
   CATEGORIES_API_URL=http://192.168.0.109:8081/api/product/categories
   ```

## External Dependencies Setup

### MongoDB Atlas

1. Create a MongoDB Atlas account
2. Create a cluster with vector search support
3. Create a database named `bizlink`
4. Create collections: `categories`, `products`
5. Enable vector search index on `categories` collection:
   - Index name: `category_vector_index`
   - Vector dimensions: 512
   - Similarity: cosine

### OpenSearch

#### Installation (Windows)

1. **Download OpenSearch**
   - Visit: https://opensearch.org/downloads.html
   - Download OpenSearch Windows zip file (e.g., opensearch-3.7.0-windows-x64.zip)
   - Extract to a location (e.g., `C:\opensearch`)

2. **Configure OpenSearch**
   - Navigate to the config directory: `C:\opensearch\config`
   - Edit `opensearch.yml`:
     ```yaml
     cluster.name: opensearch-cluster
     node.name: node-1
     network.host: 0.0.0.0
     http.port: 9200
     discovery.type: single-node
     # Disable security for development (optional)
     plugins.security.disabled: true
     ```

3. **Start OpenSearch**
   ```powershell
   cd C:\opensearch\bin
   opensearch.bat
   ```
   - OpenSearch will start on `http://localhost:9200`
   - Keep this terminal open while running the service

4. **Verify OpenSearch is running**
   ```powershell
   curl http://localhost:9200
   ```
   - Should return JSON response with cluster information

#### Installation (Linux/Mac)

1. **Download and extract**
   ```bash
   wget https://artifacts.opensearch.org/releases/bundle/opensearch/3.7.0/opensearch-3.7.0-linux-x64.tar.gz
   tar -xzf opensearch-3.7.0-linux-x64.tar.gz
   cd opensearch-3.7.0
   ```

2. **Configure and start**
   ```bash
   # Edit config/opensearch.yml as needed
   ./bin/opensearch
   ```

#### OpenSearch Dashboard (Optional)

1. **Download OpenSearch Dashboard**
   - Download from same page as OpenSearch
   - Extract to a location (e.g., `C:\opensearch-dashboards`)

2. **Configure Dashboard**
   - Navigate to config directory: `C:\opensearch-dashboards\config`
   - Edit `opensearch_dashboards.yml`:
     ```yaml
     opensearch.hosts: ["http://localhost:9200"]
     ```

3. **Start Dashboard**
   ```powershell
   cd C:\opensearch-dashboards\bin
   opensearch-dashboards.bat
   ```
   - Access Dashboard at: `http://localhost:5601`

4. **Create Index Pattern**
   - Open Dashboard in browser
   - Go to Management → Stack Management → Index Patterns
   - Create index pattern: `image-text-pdf-logs-*`
   - Select `@timestamp` as time field

#### Connection Verification

Test the connection from your service:
```powershell
# Test OpenSearch is accessible
curl http://localhost:9200/_cluster/health

# Test index creation
curl -X PUT http://localhost:9200/test-index
```

## Running the Service

### Method 1: Direct Python
```bash
python -m app.main
```

### Method 2: Uvicorn
```bash
uvicorn app.main:app --host 127.0.0.1 --port 6000
```

### Method 3: Uvicorn with reload (development)
```bash
uvicorn app.main:app --host 127.0.0.1 --port 6000 --reload
```

The service will start on `http://127.0.0.1:6000`

## API Endpoints

### Health Check
```http
GET /health
```

### Extract from Image
```http
POST /image
Content-Type: multipart/form-data

file: <image_file>
```

### Extract from PDF
```http
POST /pdf
Content-Type: multipart/form-data

file: <pdf_file>
```

### Extract from Text
```http
POST /getText
Content-Type: application/json

{
  "text": "raw text content"
}
```

### Get Products by Batch ID
```http
GET /products/image/{batch_id}
GET /products/pdf/{batch_id}
GET /products/text/{batch_id}
```

### Sync Categories
```http
POST /sync/categories
```

## File Size Limits

- Maximum file size: 15MB
- Supported image formats: JPEG, PNG, JPG, WebP, BMP
- Supported document format: PDF

## Architecture

```
Client Request
    ↓
FastAPI Service
    ↓
├── OCR Processing (PaddleOCR)
├── LLM Extraction (Gemini)
├── Normalization (Gemini)
├── Embedding (Gemini)
├── Category Matching (MongoDB Vector Search)
└── Logging (OpenSearch)
```

## Logging

Logs are sent to OpenSearch with the following structure:
- Index: `image-text-pdf-logs-YYYY.MM.DD`
- Events: `startup`, `shutdown`, `http_request`, `extract`, `pdf`, `getText`, `unhandled_exception`
- Circuit breaker pattern for resilience

## Troubleshooting

### Service fails to start
- Check MongoDB connection string in `.env`
- Verify Gemini API key is valid
- Ensure all dependencies are installed

### OCR errors
- Verify image file format is supported
- Check file size is under 15MB
- Ensure PaddleOCR is properly installed

### Category matching fails
- Verify MongoDB Atlas vector search index exists
- Check that categories are synced via `/sync/categories` endpoint
- Ensure embedding dimensions match (512)

### Logging errors
- Verify OpenSearch is running
- Check `OPENSEARCH_HOST` in `.env`
- Circuit breaker will temporarily disable logging on repeated failures

## Development

### Project Structure
```
Image-Text-Pdf-AI-Service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models.py            # Pydantic models
│   ├── database.py          # MongoDB connection
│   ├── ocr_service.py        # OCR processing
│   ├── extraction.py        # LLM extraction
│   ├── embeddings.py        # Embedding generation
│   ├── matching.py          # Category matching
│   ├── normalization.py     # Name normalization
│   ├── document_loader.py   # PDF/image loading
│   └── sync_categories.py   # Category sync
├── requirements.txt
├── .env
└── README.md
```

### Adding New Features
1. Add new endpoints in `app/main.py`
2. Create corresponding models in `app/models.py`
3. Add business logic in appropriate module files
4. Update this README with new endpoints

## License

[Prospect infoSystem inc]

---
## Support

For issues and questions, please contact
 [Phone no: +91 96527 96086, +91 62813 70532].
 [EMAIL To: thalluridhanujay@gmail.com , tumati.bhanuprakash947@gmail.com ]




