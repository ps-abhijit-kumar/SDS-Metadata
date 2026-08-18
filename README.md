# 🧠 AI Document Intelligence Platform

> A production-oriented AI-powered **Safety Data Sheet (SDS) Document Intelligence Platform** that extracts structured metadata and enables grounded conversational question answering over uploaded SDS documents using a fully local Retrieval-Augmented Generation (RAG) pipeline.

Built with **FastAPI, Streamlit, Ollama, ChromaDB, SQLite, PyMuPDF, and local embedding models**, the platform is designed to process sensitive documents without depending on external cloud AI services.

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-success?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-blueviolet)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 📌 Overview

Safety Data Sheets (SDS) contain critical information about chemical products, hazards, first-aid measures, handling and storage, exposure controls, regulatory information, and other safety requirements.

Manually extracting and searching this information from long PDF documents can be time-consuming and error-prone.

This project provides a local AI-powered solution that combines:

- Structured SDS metadata extraction
- Semantic document retrieval
- Section-aware retrieval
- Conversational RAG
- Document-grounded question answering
- Strict anti-hallucination guardrails
- Local LLM inference
- Document history and audit information

The platform allows users to upload an SDS PDF, extract the required metadata, and then ask questions about the uploaded document through a conversational interface.

---

# 🎯 Objectives

The platform was designed to:

- Extract mandatory structured metadata from SDS PDFs.
- Retrieve relevant information from large SDS documents.
- Allow users to ask natural-language questions about uploaded documents.
- Provide answers grounded only in the uploaded document.
- Prevent the LLM from answering unrelated outside-world questions.
- Reduce hallucinated responses through retrieval and grounding validation.
- Maintain document-level retrieval isolation.
- Keep sensitive documents and AI processing local.
- Provide a production-oriented backend architecture.
- Maintain extraction and document history for auditability.

---

# ✨ Core Features

## 📄 SDS Document Processing

- PDF upload
- PDF text extraction
- Text normalization
- Semantic document chunking
- SDS section detection
- Metadata extraction
- Structured metadata validation
- Document-specific vector indexing
- Multiple document history

The primary extracted metadata includes:

```text
Product Name
Company / Manufacturer
Language
Regulatory Jurisdiction
```
## 🤖 Conversational RAG Chat

The platform includes a conversational chatbot that allows users to ask questions about the currently uploaded SDS.

**Example:**

> **User:** What are the first aid measures?
>
> **A:** Provides the relevant first-aid information from the uploaded SDS.

Users can ask natural-language questions without needing to know the exact section number of the SDS.

Examples include:

- What are the first aid measures?
- What are the storage conditions?
- What precautions are listed?
- What are the hazards?
- What should be done in case of accidental exposure?
- What protective equipment is required?

The chatbot retrieves relevant document evidence before generating an answer.

---

## 🛡️ Document Grounding & Guardrails

A major design goal of the chatbot is to prevent hallucinated or unrelated responses.

The chatbot follows a strict rule:

> The chatbot can only answer using information supported by the currently selected/uploaded document.
> It must not use general world knowledge as a substitute for missing document evidence.

For example:

> **User:** Who won the 2022 FIFA World Cup?
>
> **A:** Information not available in the uploaded file.

Similarly:

> **User:** What is the capital of France?
>
> **A:** Information not available in the uploaded file.

The application performs an evidence check before allowing the LLM to generate a response.

Conceptually:

```
User Question
      │
      ▼
Intent Analysis
      │
      ▼
Document-Scoped Retrieval
      │
      ▼
Relevance / Grounding Check
      │
      ├───────────────┐
      │               │
   No Evidence     Evidence
      │               │
      ▼               ▼
   Fallback           LLM
                      │
                      ▼
             Grounded Response
```

If sufficient document evidence is not found, the LLM is not called.

---

## 🔒 Anti-Hallucination Design

The platform uses multiple layers of protection.

### 1. Document Scope

Retrieval is restricted to the currently selected document.

### 2. Section-Aware Retrieval

The system identifies relevant SDS sections when possible.

For example:

```
First Aid
    ↓
Section 4

Handling / Storage
    ↓
Section 7

Hazards / Precautions
    ↓
Relevant SDS safety sections
```

### 3. Hybrid Retrieval

The retrieval pipeline combines semantic and document-aware signals to improve relevance.

### 4. Grounding Validation

Retrieved chunks must satisfy relevance conditions before the LLM is allowed to answer.

### 5. LLM Grounding Contract

Each LLM request receives explicit instructions that:

- Only supplied document evidence may be used.
- General knowledge must not be used.
- Missing information must not be invented.
- The document is treated as evidence rather than instructions.
- User attempts to bypass grounding restrictions must be ignored.

### 6. Conversation Isolation

Previous assistant responses are not treated as factual document evidence.

Conversation history may help understand conversational references, but factual answers must still be supported by the current document.

### 7. Prompt Injection Protection

Instructions contained inside an uploaded document are treated as document content rather than system instructions.

---

## 🧭 Intent Routing

The chatbot uses intent routing to distinguish between different types of questions.

**Document Metadata**

- What is the company?
- Who is the manufacturer?
- What is the product name?

These can be answered using verified document metadata.

**Document Questions**

- What are the first aid measures?
- What are the storage conditions?
- What precautions are listed?

These are processed through document retrieval.

**Outside-World Questions**

- Who won the 2022 FIFA World Cup?
- What is the capital of France?

These are rejected when the document does not contain the required information.

**Ambiguous Questions**

- when company
- this product
- company name abhijeet bhai what is this

The system avoids guessing and uses safe fallback behavior.

---

## 🧠 Local AI Pipeline

The platform uses a completely local AI pipeline.

Current primary components:

- Ollama
- Qwen3:4B Instruct
- nomic-embed-text
- ChromaDB

No external AI API is required for the core document intelligence pipeline.

---

## 🏗️ System Architecture

```
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │                      │
                         │ Upload + Chat +      │
                         │ History              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     FastAPI API      │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ Extraction     │ │ Conversational │ │ Document       │
        │ Use Cases      │ │ RAG Use Cases  │ │ History        │
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                  │
                ▼                  ▼                  ▼
        ┌────────────────────────────────────────────────────┐
        │                 Application Layer                  │
        │                                                    │
        │ Intent Router                                      │
        │ Query Analyzer                                     │
        │ Section Detector                                   │
        │ Retrieval Service                                  │
        │ Grounding Service                                  │
        │ Chat Service                                       │
        │ Metadata Validator                                 │
        └────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │               Infrastructure Layer                 │
        │                                                    │
        │ PyMuPDF                                            │
        │ Ollama                                             │
        │ nomic-embed-text                                   │
        │ ChromaDB                                           │
        │ SQLite                                             │
        └────────────────────────────────────────────────────┘
```

---

## 🔄 Document Processing Pipeline

```
PDF Upload
    │
    ▼
Read PDF with PyMuPDF
    │
    ▼
Extract Text
    │
    ▼
Normalize Text
    │
    ▼
Detect Language
    │
    ▼
Semantic Chunking
    │
    ▼
Generate Embeddings
(nomic-embed-text)
    │
    ▼
Store Document Chunks
in ChromaDB
    │
    ▼
Retrieve Relevant Evidence
    │
    ▼
Extract Metadata
    │
    ▼
Validate Metadata
    │
    ▼
Persist Results
in SQLite
```

---

## 💬 Conversational RAG Pipeline

```
User Question
      │
      ▼
Intent Analysis
      │
      ▼
Section Detection
      │
      ▼
Document-Scoped Retrieval
      │
      ▼
Hybrid Relevance Filtering
      │
      ▼
Grounding Validation
      │
      ├───────────────┐
      │               │
      ▼               ▼
 No Evidence       Evidence Found
      │               │
      ▼               ▼
  Fallback           Prompt
      │               │
      │               ▼
      │          Local Qwen3
      │               │
      │               ▼
      │       Grounded Answer
      │               │
      └───────────────┴──────►
                         Streamlit UI
```

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13+ |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM Runtime | Ollama |
| LLM | Qwen3:4B Instruct |
| Embeddings | nomic-embed-text |
| Vector Database | ChromaDB |
| Database | SQLite |
| PDF Parsing | PyMuPDF |
| Retrieval | Semantic / Hybrid Retrieval |
| Architecture | Clean Architecture |
| Logging | Python Logging |
| Containerization | Docker-ready |

---

## 📁 Project Structure

```
SDS-Metadata/
│
├── app/
│   ├── application/
│   │   ├── services/
│   │   │   ├── chat_service.py
│   │   │   ├── grounding_service.py
│   │   │   ├── intent_router.py
│   │   │   ├── retrieval_service.py
│   │   │   ├── section_detector.py
│   │   │   └── ...
│   │   │
│   │   └── use_cases/
│   │       ├── chat_with_document_use_case.py
│   │       ├── extract_metadata_use_case.py
│   │       └── ...
│   │
│   ├── domain/
│   │   ├── entities/
│   │   ├── exceptions/
│   │   └── repositories/
│   │
│   ├── infrastructure/
│   │   ├── configuration/
│   │   ├── database/
│   │   ├── embeddings/
│   │   ├── llm/
│   │   ├── parser/
│   │   ├── retrieval/
│   │   ├── vectorstore/
│   │   └── logging/
│   │
│   ├── presentation/
│   │   ├── routers/
│   │   ├── dependencies/
│   │   └── schemas/
│   │
│   └── main.py
│
├── frontend/
│
├── tests/
│
├── docs/
│   └── images/
│
├── data/
│   ├── chroma_db/
│   └── platform.db
│
├── logs/
│
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Install:

- Python 3.13+
- Git
- Ollama

### Install Ollama

Download and install Ollama from:

https://ollama.com

Pull the required models:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

You should see the required models.

### Clone Repository

```bash
git clone https://github.com/ps-abhijit-kumar/SDS-Metadata.git
cd SDS-Metadata
```

### Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Copy:

```
.env.example
```

to:

```
.env
```

Update environment variables if required.

Typical configuration includes:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `CHROMA_DB_DIR`
- `SQLITE_DB_PATH`

### Start Ollama

Ollama normally runs as a local service.

Verify:

```bash
ollama ps
```

or:

```bash
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

### Start Backend

Open a PowerShell terminal inside the project:

```bash
.\.venv\Scripts\Activate.ps1
```

Run:

```bash
python -m uvicorn app.main:app --reload
```

Backend:

http://localhost:8000

Swagger API:

http://localhost:8000/docs

### Start Frontend

Open another PowerShell terminal:

```bash
cd SDS-Metadata
```

Activate the environment:

```bash
.\.venv\Scripts\Activate.ps1
```

Run:

```bash
python -m streamlit run frontend/app.py
```

Application:

http://localhost:8501

---

## 🔄 Example Workflow

**Step 1 — Upload an SDS**

Upload an SDS PDF through the Streamlit interface.

**Step 2 — Metadata Extraction**

The system processes the document and extracts:

- Product Name
- Company / Manufacturer
- Language
- Regulatory Jurisdiction

**Step 3 — Document Indexing**

The PDF is:

```
Extracted
    ↓
Normalized
    ↓
Chunked
    ↓
Embedded
    ↓
Stored in ChromaDB
```

**Step 4 — Ask Questions**

The user can then ask questions about the uploaded SDS.

Example:

> What are the first aid measures?

The system retrieves the relevant evidence and generates a grounded answer.

**Step 5 — Grounding Protection**

If the document does not contain the requested information:

> Information not available in the uploaded file.

The system does not use outside knowledge to fill the gap.

---

## 📋 Example Chat Behavior

**Document-grounded question**

> **User:** What are the first aid measures?
>
> **A:** Provides the first-aid information found in the uploaded SDS.

**Another document-grounded question**

> **User:** What are the storage conditions?
>
> **A:** Provides storage information supported by the SDS.

**Outside-world question**

> **User:** Who won the 2022 FIFA World Cup?
>
> **A:** Information not available in the uploaded file.

**General knowledge question**

> **User:** What is the capital of France?
>
> **A:** Information not available in the uploaded file.

**Document metadata question**

> **User:** What is the company?
>
> **A:** Returns the verified manufacturer/company from the uploaded SDS.

---

## 📡 API Endpoints

The application exposes FastAPI endpoints for document processing and conversational RAG.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application / health status |
| GET | `/health` | Health check |
| POST | `/api/v1/extract` | Upload and extract SDS metadata |
| GET | `/api/v1/documents` | List processed documents |
| POST | `/api/v1/chat/stream` | Streaming document-grounded chat |
| GET | `/docs` | Swagger API documentation |

Additional endpoints may be available depending on the current API configuration.

---

## ⚙️ Configuration

The application uses environment variables.

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Local LLM model |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model |
| `CHROMA_DB_DIR` | ChromaDB storage directory |
| `SQLITE_DB_PATH` | SQLite database path |

Typical local Ollama configuration:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

## 📊 Document History & Audit

The application maintains document processing history using SQLite.

The system records information related to:

- Uploaded documents
- Extraction status
- Processing results
- Document metadata
- Processing events
- Errors
- Performance information

This provides an audit-friendly foundation for document intelligence workflows.

---

## 📝 Logging

Application logs are maintained under:

```
logs/
```

Typical log information includes:

- Application startup
- Document uploads
- PDF extraction
- Language detection
- Semantic chunking
- Embedding generation
- ChromaDB operations
- Retrieval
- Grounding decisions
- LLM calls
- Metadata validation
- Chat routing
- Errors
- Performance metrics

The chat pipeline also provides visibility into whether evidence was retrieved and whether the LLM was invoked.

---

## 🔐 Security & Privacy

The platform follows a local-first architecture.

Key characteristics:

- No external AI API is required.
- Documents remain on the local machine.
- LLM inference runs through local Ollama.
- Embeddings are generated locally.
- Vector storage is local.
- SQLite persistence is local.
- Document retrieval is scoped to the selected document.
- Uploaded document content is treated as untrusted data.
- Prompt injection attempts inside documents are not treated as system instructions.

This architecture is suitable for environments where document privacy and local processing are important.

---

## 🧪 Testing

The repository contains testing infrastructure.

Testing should cover areas such as:

- Metadata extraction
- Retrieval
- Grounding
- Intent routing
- Section detection
- Conversational RAG
- Document isolation
- API behavior

Example:

```bash
pytest
```

Verbose:

```bash
pytest -v
```

End-to-end verification can also be performed according to the project's available verification scripts.

---

## ⚡ Performance Characteristics

The current architecture supports:

- Local LLM inference
- Local embeddings
- Batched embedding generation
- Semantic retrieval
- Section-aware retrieval
- Hybrid retrieval
- Async document extraction
- ChromaDB persistence
- SQLite persistence
- Streaming conversational responses
- Multiple document history

LLM inference time depends on:

- Document size
- Number of retrieved chunks
- CPU/GPU resources
- Ollama model
- Prompt size

Local inference can take significantly longer than cloud inference on CPU-only systems. This is an expected trade-off for local processing and document privacy.

---

## 🏆 Engineering Highlights

This project demonstrates:

- Clean Architecture
- Dependency Injection
- Repository Pattern
- Retrieval-Augmented Generation
- Local LLM Integration
- Vector Database Design
- Semantic Search
- Hybrid Retrieval
- Section-Aware Retrieval
- Grounding Validation
- Intent Routing
- Prompt Engineering
- Anti-Hallucination Guardrails
- Prompt Injection Protection
- Async Processing
- Streaming Responses
- Structured Logging
- SQLite Persistence
- Environment-Based Configuration
- Modular Enterprise Project Structure
- Git-based version control

---

## 💡 Engineering Challenges & Solutions

| Challenge | Solution |
|---|---|
| Large SDS documents | Semantic chunking before retrieval |
| Irrelevant retrieval | Hybrid relevance filtering |
| Hallucinated responses | Retrieval + grounding validation |
| Outside-world questions | Evidence gate before LLM invocation |
| Long conversations | Limited conversational context |
| LLM context loss | Per-request grounding contract |
| Document prompt injection | Treat document text as untrusted evidence |
| Ambiguous metadata queries | Intent routing |
| Section-specific questions | Section-aware retrieval |
| Cloud dependency | Local Ollama deployment |
| Slow embeddings | Batched embedding requests |
| Long extraction pipeline | Async processing |
| Configuration management | Environment-based configuration |
| Maintainability | Clean Architecture and dependency injection |

---

## ⚡ Performance Optimizations

Current optimizations include:

- Batched embedding generation
- Persistent ChromaDB storage
- Document-scoped retrieval
- Section-aware retrieval
- Hybrid retrieval
- Async extraction
- Local inference
- Reusable application dependencies
- SQLite persistence
- Structured logging

---

## 📸 Application Screenshots

- Home
- Upload Document
- Extraction Result
- Processing History
- Conversational RAG
- Swagger API

Add the current chat interface screenshot here:

```
docs/images/chat.png
```

---

## 📊 Current Capabilities

| Feature | Status |
|---|---|
| SDS PDF Processing | ✅ |
| Metadata Extraction | ✅ |
| Product Name Extraction | ✅ |
| Company / Manufacturer Extraction | ✅ |
| Language Detection | ✅ |
| Regulatory Jurisdiction Detection | ✅ |
| Semantic Chunking | ✅ |
| Local Embeddings | ✅ |
| ChromaDB Vector Storage | ✅ |
| Section-Aware Retrieval | ✅ |
| Hybrid Retrieval | ✅ |
| Conversational RAG | ✅ |
| Streaming Chat | ✅ |
| Document Grounding | ✅ |
| Anti-Hallucination Guardrails | ✅ |
| Outside-World Question Rejection | ✅ |
| Document-Scoped Retrieval | ✅ |
| Metadata Intent Routing | ✅ |
| Prompt Injection Protection | ✅ |
| SQLite Persistence | ✅ |
| Processing History | ✅ |
| FastAPI Backend | ✅ |
| Streamlit Frontend | ✅ |
| Local Ollama Inference | ✅ |
| Docker Support | ✅ |

---

## 📌 Known Limitations

Current version focuses primarily on SDS documents.

Current limitations include:

- SDS-focused document processing
- Local deployment architecture
- LLM performance depends on local hardware
- OCR support for scanned PDFs is not currently the primary processing path
- Retrieval quality depends on document text extraction quality
- Very large documents may require additional retrieval optimization
- Multilingual behavior depends on the document and local model capabilities

---

## 🚀 Future Roadmap

### Version 1.1

- Improved retrieval evaluation
- Advanced metadata validation
- Better multilingual RAG support
- Enhanced source attribution
- Improved confidence scoring
- API versioning
- Expanded document analytics

### Version 1.2

- Batch PDF processing
- Multi-document conversational retrieval
- CSV / Excel export
- Background task queue
- Performance dashboard
- Advanced audit reporting

### Version 2.0

- Multi-user authentication
- PostgreSQL support
- Redis caching
- Kubernetes deployment
- CI/CD pipeline
- Prometheus / Grafana monitoring
- Optional cloud deployment
- OCR for scanned PDFs
- Support for additional document types

---

## 🤝 Contributing

Contributions are welcome.

If you find bugs, have feature ideas, or would like to improve the platform:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Commit your changes.
5. Push the branch.
6. Open a Pull Request.

---

## 🌿 Git Development & Recovery

The project uses Git for version control.

A manually verified production checkpoint is maintained for safe recovery.

Current verified checkpoint:

- **Commit:** `03361e2`
- **Tag:** `SDS-Metadata-guardrails-working`
- **Branch:** `verified-guardrails`

This checkpoint represents the manually verified working state of the document-grounded chatbot and its guardrails.

Future development should be performed through separate commits/branches so that changes can be safely reverted if a regression occurs.

---

## 📄 License

This project is released under the MIT License.

---

## 👨‍💻 Author

**Abhijit Kumar**
AI Engineering Student

GitHub: [https://github.com/ps-abhijit-kumar](https://github.com/ps-abhijit-kumar)

---

## 🙏 Acknowledgements

This project leverages the excellent work of the open-source community, including:

- FastAPI
- Streamlit
- Ollama
- ChromaDB
- PyMuPDF
- SQLite
- Qwen
- Nomic embedding models

Special thanks to the maintainers and contributors of these projects for enabling local AI application development.

---

## ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the repository
- 💡 Share feedback
- 🤝 Contribute improvements
