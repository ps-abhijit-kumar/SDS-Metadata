# AI Document Intelligence Platform

**Production-grade local RAG pipeline for SDS (Safety Data Sheet) document metadata extraction.**

A complete, enterprise-architecture AI system that runs **100% locally** — powered by Ollama + Qwen3:8B + ChromaDB + LangChain + FastAPI + Streamlit.

## Features

- ✅ **Local-first design** — No cloud APIs, no API keys, no internet dependency
- ✅ **Complete RAG pipeline** — PDF → extract → clean → chunk → embed → retrieve → LLM → validate → store
- ✅ **Clean Architecture** — Domain, Application, Infrastructure, Presentation layers with full SOLID compliance
- ✅ **Production-ready** — Structured logging, exception handling, validation, Docker
- ✅ **Modular & testable** — 100% dependency injection, abstract interfaces, comprehensive test suite
- ✅ **Multi-document batch processing** — Upload one or many PDFs in a single request
- ✅ **RESTful FastAPI backend** — Full API documentation at `/docs`
- ✅ **Streamlit web interface** — Beautiful, responsive UI for document upload and result viewing

## What It Does

Upload an SDS PDF document. The system automatically extracts:

1. **Product Name** — The commercial or trade name of the chemical product
2. **Language** — The language the document is written in (English, Portuguese, Spanish, German, French, etc.)
3. **Jurisdiction** — The regulatory framework the SDS complies with:
   - United States (OSHA / HazCom 2012)
   - Canada (WHMIS 2015)
   - European Union (REACH / CLP)
   - United Kingdom (UK REACH)
   - Brazil (ABNT NBR 14725)
   - Mexico (NOM-018-STPS)
   - Australia, New Zealand, Japan, China, South Korea, India, Singapore
   - Or any other global SDS standard

**Language and Jurisdiction are independent.** For example:
- An English document following Brazilian regulations
- A German document following EU regulations
- A Portuguese document following Portugal/Brazil regulations

## Tech Stack

- **Python 3.12+**
- **FastAPI 0.139** — Type-safe, async web framework
- **Streamlit 1.45** — Web UI framework
- **Ollama** — Local LLM runtime
- **Qwen3:8B** — Default open-source LLM (can swap for Llama2, Mistral, etc.)
- **LangChain 0.3** — RAG orchestration
- **ChromaDB 1.0** — Vector database
- **nomic-embed-text:latest** — Embedding model
- **SQLite** — Document metadata store
- **PyMuPDF** — PDF text extraction
- **Pydantic v2** — Data validation
- **pytest** — Testing framework
- **Docker** — Containerisation

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Ollama** installed locally (download from https://ollama.ai)
- **Git**

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SDS-Metadata-Extractor
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Ollama Models

In a separate terminal, start Ollama and pull the required models:

```bash
ollama serve

# In another terminal:
ollama pull qwen3:8b
ollama pull nomic-embed-text:latest
```

Verify the models are available:

```bash
ollama list
```

### 5. Configure Environment

Copy the example config and adjust if needed:

```bash
cp .env.example .env
```

The defaults assume Ollama is running at `http://127.0.0.1:11434` and the database is at `./data/platform.db`. Adjust if your setup differs.

### 6. Initialize Database

```bash
python scripts/init_db.py
```

### 7. Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to see the API documentation.

### 8. Start the Streamlit Frontend (in a new terminal)

```bash
streamlit run frontend/app.py
```

Open http://127.0.0.1:8501 in your browser.

### 9. Upload & Extract

1. Go to the **Extract** tab
2. Upload one or more SDS PDF files
3. Click "Extract Metadata"
4. Wait for the RAG pipeline to complete (typically 1–3 minutes per document depending on LLM inference speed)
5. View results — product name, language, jurisdiction

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌──────────────────┐                   ┌──────────────────────┐│
│  │  Streamlit UI    │                   │   FastAPI Backend    ││
│  │  (frontend/      │◄──────HTTP───────►│   (app/presentation) ││
│  │   app.py)        │                   │                      ││
│  └──────────────────┘                   └──────────────────────┘│
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Dependency Container     │
                    │  (DI + Lifespan)          │
                    └────────────┬──────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────────┐  ┌────────▼──────────┐  ┌─────────▼──────┐
│Application Layer   │  │Infrastructure     │  │Domain Layer    │
│                    │  │                    │  │                │
│• Use Cases         │  │• SQLite Database   │  │• Entities      │
│• Services          │  │• Ollama LLM        │  │• Value Objects │
│• DTOs              │  │• Embeddings        │  │• Exceptions    │
│• Interfaces        │  │• Vector Store      │  │• Repositories  │
│• Validators        │  │• PDF Reader        │  │• Enums         │
│• Prompt Builder    │  │• Config & Logging  │  │                │
└────────────────────┘  └────────────────────┘  └────────────────┘
```

### Pipeline Flow

```
Upload PDF
    ↓
Validate + Save file
    ↓
Extract raw text (PyMuPDF)
    ↓
Clean & normalise
    ↓
Semantic chunking (detect SDS sections)
    ↓
Generate embeddings (nomic-embed-text)
    ↓
Store in ChromaDB
    ↓
Multi-query retrieval (8 different retrieval queries)
    ↓
Build structured prompt
    ↓
Call Ollama LLM (Qwen3:8B)
    ↓
Parse & validate LLM response
    ↓
Persist to SQLite
    ↓
Return results
```

## API Endpoints

### POST `/api/v1/extract`
Upload one or more SDS PDF files and extract metadata.

**Request:** Multipart file upload
**Response:**
```json
{
  "total": 2,
  "results": [
    {
      "document_id": "uuid",
      "filename": "sds_sample.pdf",
      "status": "completed",
      "product_name": "Aceton 99%",
      "language": "English",
      "jurisdiction": "United States (OSHA / HazCom 2012)",
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "document_id": "uuid",
      "filename": "invalid_doc.txt",
      "status": "failed",
      "error_message": "File type '.txt' is not supported. Only PDF files are accepted."
    }
  ]
}
```

### GET `/api/v1/documents`
List all processed documents.

### GET `/api/v1/documents/{document_id}`
Get a single document's extraction result.

### DELETE `/api/v1/documents/{document_id}`
Delete a document record and its associated vector store entries.

### GET `/health`
Application health check.

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Unit Tests Only

```bash
pytest tests/unit/ -v
```

### Run Integration Tests Only

```bash
pytest tests/integration/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov-report=html
```

## Docker Deployment

### Build & Run with Docker Compose

```bash
docker-compose up --build
```

This starts three services:
- **ollama** — Local LLM runtime on port 11434
- **api** — FastAPI backend on port 8000
- **frontend** — Streamlit on port 8501

Access:
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Ollama API: http://localhost:11434

## Configuration

All configuration is environment-based. See `.env.example` for all available options:

```env
# Application
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
DEBUG=false

# File upload
UPLOAD_DIR=./data/uploads
UPLOAD_MAX_SIZE_MB=50

# Database
DATABASE_URL=sqlite:///./data/platform.db

# Vector store
CHROMA_DB_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=documents

# Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_LLM_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_TIMEOUT_SECONDS=180

# RAG tuning
CHUNK_SIZE=600
CHUNK_OVERLAP=100
RETRIEVAL_K=8
EMBEDDING_BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
```

## Project Structure

```
app/
├── domain/                      # Pure business logic, no framework dependencies
│   ├── entities/
│   ├── value_objects/
│   ├── repositories/            # Abstract interfaces
│   ├── exceptions/
│   └── enums/
├── application/                 # Use cases, services, DTOs
│   ├── use_cases/
│   ├── services/
│   └── dto/
├── infrastructure/              # Implementation of interfaces
│   ├── database/
│   ├── repositories/
│   ├── llm/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── pdf/
│   ├── logging/
│   └── configuration/
├── presentation/                # FastAPI routers, schemas, middleware
│   ├── routers/
│   ├── schemas/
│   ├── dependencies/            # DI container
│   └── middleware/
└── main.py

frontend/
└── app.py                        # Streamlit UI

tests/
├── unit/                         # Unit tests for services, validators, etc.
└── integration/                  # Integration tests for database, repositories

scripts/
└── init_db.py                    # Database initialization

docker-compose.yml               # Multi-container orchestration
Dockerfile                        # FastAPI container definition
requirements.txt                  # Python dependencies
pyproject.toml                    # Build config & pytest settings
.env.example                      # Configuration template
README.md                         # This file
```

## Performance Notes

- **PDF extraction**: ~1–2 seconds per page using PyMuPDF
- **Embedding generation**: ~10–30 seconds depending on chunk count (batched, uses Ollama)
- **LLM inference**: ~30–90 seconds depending on model and system (Qwen3:8B)
- **Total per document**: ~2–3 minutes on average hardware

Optimizations included:
- Batch embedding generation (32 texts per batch by default)
- Multi-query retrieval for better recall
- Token limit on LLM output (512 tokens max for speed)
- Temperature = 0 (deterministic extraction)
- ChromaDB similarity search with deduplication

## Troubleshooting

### "Cannot connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Verify models are available: `ollama list`
- Check `OLLAMA_BASE_URL` in `.env` points to the correct Ollama host

### "No such file or directory: ./data/platform.db"
- Run `python scripts/init_db.py` to create the database

### "PDF extraction failed: Cannot open PDF"
- Ensure the uploaded file is a valid, non-corrupted PDF
- Try opening the file in a PDF reader to verify

### "LLM response did not contain any expected metadata fields"
- The LLM output format was unexpected
- Try with a different document or adjust the prompt in `app/application/services/prompt_builder.py`

### "No relevant chunks were retrieved"
- The semantic chunking or retrieval failed
- Try adjusting `RETRIEVAL_K` or `CHUNK_SIZE` in `.env`

## Contributing

This is a production-ready reference implementation. Feel free to:
- Add support for additional document types (MSDS, GHS documents, etc.)
- Implement different LLM backends (Llama2, Mistral, etc.)
- Add advanced retrieval strategies (re-ranking, query expansion, etc.)
- Extend metadata extraction to capture additional fields
- Implement batch processing with async workers

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, feature requests, or questions, please refer to the project documentation or open an issue.

---

**Built for enterprise. Runs locally. Zero external dependencies.**
