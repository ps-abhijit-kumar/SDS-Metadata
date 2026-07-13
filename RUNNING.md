# AI Document Intelligence Platform – Running

## Status: ✓ OPERATIONAL

The complete production-grade SDS Metadata Extraction System is now **running and fully functional**.

---

## Services Active

### 📊 FastAPI Backend (API)
- **URL:** http://127.0.0.1:8000
- **Health Check:** http://127.0.0.1:8000/health
- **API Documentation:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Status:** ✓ Running on Uvicorn

### 🎨 Streamlit Frontend (UI)
- **URL:** http://127.0.0.1:8501
- **Status:** ✓ Running

---

## How to Use

### Step 1: Open the Web Interface
Go to: **http://127.0.0.1:8501**

### Step 2: Upload an SDS Document
1. Click the file uploader
2. Select a PDF Safety Data Sheet document
3. Click "Upload"

### Step 3: Extract Metadata
The system will automatically:
1. ✓ Extract text from the PDF
2. ✓ Parse and clean the document
3. ✓ Detect SDS sections (16 patterns)
4. ✓ Perform semantic chunking
5. ✓ Generate embeddings (nomic-embed-text)
6. ✓ Store in ChromaDB vector database
7. ✓ Retrieve relevant chunks via semantic search
8. ✓ Query Ollama Qwen3:8B LLM
9. ✓ Extract and validate metadata
10. ✓ Persist to SQLite database
11. ✓ Display results

### Step 4: View Results
The extracted metadata will be displayed:
- **File ID** – Unique document identifier
- **Product Name** – Chemical/material product name
- **Language** – Document language (English, Portuguese, Spanish, etc.)
- **Jurisdiction** – Regulatory framework (OSHA, REACH, ABNT NBR 14725, etc.)

---

## Technology Stack

### Backend
- **Python** 3.12+
- **FastAPI** 0.115.9
- **Uvicorn** (ASGI Server)
- **Pydantic** V2 (Validation)
- **pytest** (Testing)

### Frontend
- **Streamlit** (Web UI)
- **HTTP Client** (API Integration)

### AI/ML
- **Ollama** (Local LLM Runtime)
- **Qwen3:8B** (Large Language Model)
- **LangChain** 0.3.25 (RAG Orchestration)
- **ChromaDB** 1.5.9 (Vector Database)
- **nomic-embed-text** (Embeddings Model)

### Data
- **SQLite** (Metadata DB)
- **WAL Mode** (Concurrent Access)
- **ChromaDB** (Vector Search)

### Document Processing
- **PyMuPDF** (PDF Text Extraction)
- **Text Cleaning** (Normalization, Deduplication)
- **Semantic Chunking** (16 SDS Section Patterns)

### Infrastructure
- **Docker** (Containerization)
- **python-dotenv** (Configuration Management)
- **Python Logging** (Audit Trail)

---

## Architecture

### Clean Architecture (SOLID Principles)
```
┌─────────────────────────────────────┐
│   Presentation Layer                │
│   FastAPI Routes + Streamlit UI     │
├─────────────────────────────────────┤
│   Application Layer                 │
│   Use Cases + Services              │
├─────────────────────────────────────┤
│   Domain Layer                      │
│   Entities + Value Objects          │
├─────────────────────────────────────┤
│   Infrastructure Layer              │
│   Repositories + External Clients   │
└─────────────────────────────────────┘
```

### RAG Pipeline (10 Stages)
1. **PDF Validation** – Verify file integrity
2. **Text Extraction** – Extract text from PDF pages
3. **Text Normalization** – Clean, deduplicate, remove noise
4. **SDS Section Detection** – Identify 16 standard sections
5. **Semantic Chunking** – Split by section and semantics
6. **Embedding Generation** – Convert text to vectors
7. **Vector Store Ingestion** – Store in ChromaDB
8. **Semantic Retrieval** – Find top-5 relevant chunks
9. **Ollama LLM Query** – Send chunks + prompt to Qwen3:8B
10. **Metadata Extraction & Validation** – Parse and store results

---

## Configuration

### Working Directory
```
c:\Coding\Projects\SDS-Metadata-Extractor
```

### Database Location
```
data\platform.db (SQLite with WAL mode)
```

### Vector Store Location
```
data\chroma_db (ChromaDB)
```

### Logs Location
```
logs\*.log (Python logging)
```

### LLM Configuration
- **Model:** qwen3:8b
- **Base URL:** http://127.0.0.1:11434 (Ollama)
- **Timeout:** 180 seconds
- **Temperature:** 0 (deterministic)
- **Max Tokens:** 512

### Embedding Configuration
- **Model:** nomic-embed-text:latest
- **Base URL:** http://127.0.0.1:11434
- **Batch Size:** 32 texts

### RAG Configuration
- **Retrieval K:** 5 chunks
- **Chunk Size:** 1024 characters
- **Chunk Overlap:** 256 characters
- **Min Chunk Length:** 50 characters

---

## Supported Jurisdictions

The system can identify and extract metadata for SDS documents following these regulations:

- **OSHA** – United States (29 CFR 1910.1200)
- **WHMIS** – Canada (Workplace Hazardous Materials Information System)
- **REACH/CLP** – European Union (Registration, Evaluation, Authorisation of Chemicals)
- **UK REACH** – United Kingdom
- **ABNT NBR 14725** – Brazil (Associação Brasileira de Normas Técnicas)
- **Work Health and Safety Act** – Australia
- **Health and Safety at Work Act** – New Zealand
- **COSHH** – United Kingdom (Control of Substances Hazardous to Health)
- **JIS Z 7650** – Japan (Japanese Industrial Standards)
- **GB 16483** – China (Chinese Standards)
- **KCSG / MHWSE** – South Korea
- **MSIHC Audit** – India
- **SS 638: Part 1** – Singapore
- **NOM-018-STPS** – Mexico

---

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "ok", "service": "AI Document Intelligence Platform"}
```

### List Documents
```
GET /api/v1/documents
Response: [{"id": "...", "filename": "...", "uploaded_at": "...", "metadata": {...}}]
```

### Upload & Extract
```
POST /api/v1/extract
Multipart Form Data:
  - file: (PDF file)
Response: 
{
  "document_id": "...",
  "filename": "...",
  "metadata": {
    "file_id": "...",
    "product_name": "...",
    "language": "...",
    "jurisdiction": "..."
  },
  "status": "success",
  "processing_time_seconds": ...
}
```

### Get Document Metadata
```
GET /api/v1/documents/{document_id}
Response: Document metadata
```

### Delete Document
```
DELETE /api/v1/documents/{document_id}
Response: {"status": "deleted"}
```

---

## Logs

All application logs are written to `logs/` directory:
- `app.log` – Complete application log
- `extraction.log` – Extraction pipeline events
- `llm.log` – LLM interactions

View logs:
```bash
tail -f logs/app.log
```

---

## Testing

Run the test suite:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_extraction.py -v
```

Run with coverage:
```bash
pytest --cov=app tests/
```

---

## Development Commands

### Start Backend Only
```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend Only
```bash
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

### Initialize Database
```bash
.venv\Scripts\python.exe scripts/init_db.py
```

### Run Tests
```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Troubleshooting

### Ollama Not Responding
Ensure Ollama is running:
```bash
ollama serve
```

In another terminal, pull required models:
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### ChromaDB Connection Issues
Delete and reinitialize:
```bash
rmdir /s /q data\chroma_db
.venv\Scripts\python.exe scripts/init_db.py
```

### Port Already in Use
If port 8000 or 8501 is already in use, change in:
- Backend: `app/main.py` startup command
- Frontend: `frontend/app.py` or streamlit config

### Database Locked
SQLite WAL mode handles concurrent access. If you see "database is locked":
1. Ensure only one FastAPI instance is running
2. Restart the backend

---

## Next Steps

1. **Upload Test Documents** – Try with multiple SDS files
2. **Monitor Performance** – Check logs and response times
3. **Verify Accuracy** – Ensure metadata extraction is correct
4. **Configure Chunk Size** – Adjust in `app/infrastructure/configuration/settings.py`
5. **Deploy to Production** – Use Docker (see `Dockerfile`)

---

## Production Deployment

### Docker Build
```bash
docker build -t sds-extractor .
```

### Docker Run
```bash
docker run -p 8000:8000 -p 8501:8501 \
  -v /path/to/data:/app/data \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  sds-extractor
```

### Docker Compose
```bash
docker-compose up -d
```

---

## Support & Documentation

- **API Docs:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Code:** `c:\Coding\Projects\SDS-Metadata-Extractor`
- **Tests:** `c:\Coding\Projects\SDS-Metadata-Extractor\tests`

---

## Version Information

- **Python:** 3.12+
- **FastAPI:** 0.115.9
- **Streamlit:** Latest stable
- **LangChain:** 0.3.25
- **ChromaDB:** 1.5.9
- **Ollama:** Latest (ensure qwen3:8b and nomic-embed-text are pulled)
- **PyMuPDF:** Latest stable

---

**Last Updated:** 2026-07-13  
**Status:** ✓ Production Ready  
**Environment:** Development (localhost)
