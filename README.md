<div align="center">

# 🧪 SDS Metadata Extraction System

### AI-Powered Document Intelligence for Safety Data Sheets

<p>

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Qwen3-8B-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Database-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite"/>
<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge"/>

</p>

**Local AI • Retrieval-Augmented Generation • Clean Architecture • 100% Offline**

</div>

---

# Overview

SDS Metadata Extraction System is an AI-powered document intelligence application that automatically extracts structured metadata from **Safety Data Sheet (SDS)** PDF documents.

The application uses a Retrieval-Augmented Generation (RAG) pipeline running completely on your local machine. Instead of processing an entire document with a Large Language Model, it retrieves only the most relevant sections before asking the model to extract metadata. This approach improves accuracy, reduces hallucinations, and keeps inference efficient.

The project is built using **FastAPI**, **Streamlit**, **Ollama**, **ChromaDB**, and **SQLite**, following **Clean Architecture** to keep the code modular, maintainable, and easy to extend.

No cloud services.

No API keys.

No internet connection is required after downloading the models.

---

# Features

- Upload one or multiple SDS PDF documents
- Automatic PDF validation
- Text extraction using PyMuPDF
- Text cleaning and normalization
- Semantic chunking
- Local embedding generation
- ChromaDB vector search
- Retrieval-Augmented Generation (RAG)
- Local inference using Ollama
- SQLite metadata storage
- Processing history
- Structured logging
- Clean Architecture
- REST API
- Streamlit interface

---

# Extracted Metadata

The application extracts exactly four metadata fields.

| Field | Description |
|-------|-------------|
| 🌐 Language | Detects the language used in the SDS document. |
| 🌍 Jurisdiction | Identifies the applicable regulations, primarily from Section 15. |
| 🏢 Company Name | Extracts the manufacturer or supplier. |
| 🧪 Product Name | Extracts the product name, primarily from Section 1. |

If a value cannot be determined confidently, the field is returned empty instead of generating inaccurate information.

---

# What This Project Is

- AI-powered document intelligence application
- Retrieval-Augmented Generation (RAG)
- Metadata extraction system
- Local AI application
- Production-ready backend
- Modern Python project

---

# What This Project Is Not

- Chatbot
- PDF summarizer
- Question-answering system
- General document search engine

The application has one responsibility:

**Extract structured metadata from Safety Data Sheets.**

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12+ |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM Runtime | Ollama |
| LLM | Qwen3:8B |
| Embeddings | nomic-embed-text |
| Vector Database | ChromaDB |
| Database | SQLite |
| PDF Processing | PyMuPDF |
| Validation | Pydantic V2 |
| Testing | pytest |
| Configuration | python-dotenv |

---

# Architecture

The project follows **Clean Architecture**.

```
Presentation
      │
      ▼
Application
      │
      ▼
Domain
      │
      ▼
Infrastructure
```

Each layer has a single responsibility.

Business logic is completely independent of frameworks and external services.

---

# Project Workflow

```
Upload PDF
      │
      ▼
Extract Text
      │
      ▼
Clean Text
      │
      ▼
Semantic Chunking
      │
      ▼
Generate Embeddings
      │
      ▼
Store in ChromaDB
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Build Prompt
      │
      ▼
Qwen3 (Ollama)
      │
      ▼
Validate Metadata
      │
      ▼
Store in SQLite
      │
      ▼
Return Result
```

---

# Screenshots

> Screenshots will be added after the UI is finalized.

| Home | Extraction | History |
|------|------------|----------|
| Coming Soon | Coming Soon | Coming Soon |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ps-abhijit-kumar/SDS-Metadata.git

cd SDS-Metadata
```

## Create Virtual Environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Download Models

```bash
ollama pull qwen3:8b

ollama pull nomic-embed-text
```

## Configure Environment

```bash
copy .env.example .env
```

Update the required settings if necessary.

---

# Running the Application

### Terminal 1

```bash
ollama serve
```

### Terminal 2

```bash
uvicorn app.main:app --reload
```

Backend

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

### Terminal 3

```bash
streamlit run frontend/app.py
```

Frontend

```
http://127.0.0.1:8501
```
# Using the Application

1. Open the Streamlit application.

```
http://127.0.0.1:8501
```

2. Upload one or more SDS PDF documents.

3. Click **Extract Metadata**.

4. The application automatically performs:

- PDF validation
- Text extraction
- Text cleaning
- Semantic chunking
- Embedding generation
- Vector storage
- Semantic retrieval
- Prompt generation
- LLM inference
- Metadata validation
- SQLite storage

5. View the extracted metadata on the screen.

Example:

```text
language: English

jurisdiction: European Union (REACH)

company name: Thermo Fisher Scientific

product name: Optima LC/MS Acetic Acid
```

---

# REST API

Base URL

```
/api/v1
```

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/extract` | Upload SDS PDFs and extract metadata |
| GET | `/documents` | List processed documents |
| GET | `/documents/{id}` | Get metadata for a document |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/health` | Backend health check |

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# Project Structure

```
SDS-Metadata/

├── app/
│   ├── presentation/
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   └── shared/
│
├── frontend/
├── tests/
├── docs/
├── scripts/
├── data/
│
├── requirements.txt
├── pyproject.toml
├── README.md
├── .env.example
└── .gitignore
```

---

# RAG Pipeline

The application follows a Retrieval-Augmented Generation workflow.

```
SDS PDF
    │
    ▼
Extract Text
    │
    ▼
Clean Text
    │
    ▼
Semantic Chunking
    │
    ▼
Generate Embeddings
    │
    ▼
Store Vectors
    │
    ▼
Retrieve Relevant Chunks
    │
    ▼
Build Prompt
    │
    ▼
Qwen3 (Ollama)
    │
    ▼
Validate Response
    │
    ▼
Store Metadata
```

Only the retrieved context is sent to the language model.

The complete PDF is never sent to the LLM.

---

# Design Principles

The project follows modern software engineering practices.

- Clean Architecture
- SOLID Principles
- Dependency Injection
- Repository Pattern
- Factory Pattern
- Strategy Pattern
- Adapter Pattern
- DTO Pattern
- Use Case Pattern
- DRY
- KISS
- YAGNI

These principles make the project modular, testable, and easy to extend.

---

# Data Storage

## SQLite

Stores:

- Upload history
- Processing history
- Extracted metadata
- Application settings
- Statistics

---

## ChromaDB

Stores:

- Embeddings
- Semantic chunks
- Chunk metadata
- Vector index

---

# Configuration

All configuration is managed through `.env`.

Examples include:

- Upload directory
- Database path
- ChromaDB path
- Ollama URL
- LLM model
- Embedding model
- Chunk size
- Retrieval settings
- Logging level

No configuration values are hardcoded.

---

# Logging

The application uses structured logging throughout the processing pipeline.

Logs include:

- Application logs
- Error logs
- Audit logs
- Console logs

Each processing stage records useful information for debugging and monitoring.

---

# Error Handling

The project defines custom exceptions for different components.

Examples include:

- PDF Exception
- Validation Exception
- Embedding Exception
- Vector Store Exception
- LLM Exception
- Configuration Exception

This provides meaningful error messages while keeping the business logic clean.

---

# Testing

Run all tests

```bash
pytest
```

Run unit tests

```bash
pytest tests/unit
```

Run integration tests

```bash
pytest tests/integration
```

Run with coverage

```bash
pytest --cov=app
```

---

# Future Improvements

Planned enhancements include:

- OCR support for scanned PDFs
- Hybrid retrieval
- Better multilingual language detection
- Confidence scores
- Batch processing improvements
- CSV and Excel export
- Authentication
- Dashboard analytics
- Background processing
- Additional LLM providers

The architecture allows these features to be added without changing the core business logic.
# Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/your-feature
```

3. Commit your changes.

```bash
git commit -m "Add your feature"
```

4. Push the branch.

```bash
git push origin feature/your-feature
```

5. Open a Pull Request.

Before submitting a Pull Request, please ensure that:

- The project builds successfully.
- Existing tests pass.
- New code follows the project architecture.
- Documentation is updated if required.
- Business logic remains independent of infrastructure.

---

# Roadmap

## Version 1.1

- Improve metadata extraction accuracy
- Optimize semantic retrieval
- Reduce LLM response time
- Improve multilingual language detection
- Better jurisdiction extraction

---

## Version 1.2

- OCR support for scanned SDS PDFs
- Batch processing optimization
- CSV export
- Excel export
- Confidence score for extracted metadata

---

## Version 2.0

- Support additional document formats
- Hybrid retrieval (semantic + keyword)
- Multiple embedding models
- Multiple LLM providers
- Processing dashboard
- User authentication

---

# Performance

The overall processing time depends on:

- PDF size
- Number of pages
- Hardware specifications
- Selected LLM model

Typical processing flow:

| Stage | Typical Time |
|--------|--------------|
| PDF Extraction | 1–3 sec |
| Text Cleaning | <1 sec |
| Chunking | 1–2 sec |
| Embedding Generation | 5–20 sec |
| Vector Retrieval | <1 sec |
| LLM Inference | Hardware dependent |
| SQLite Storage | <1 sec |

---

# Supported Technologies

## AI

- Ollama
- Qwen3:8B
- nomic-embed-text

## Backend

- FastAPI
- Pydantic
- Python

## Frontend

- Streamlit

## Storage

- SQLite
- ChromaDB

## Processing

- PyMuPDF

## Testing

- pytest

---

# Why This Project?

This project was built to demonstrate practical AI engineering using modern software development practices.

It combines:

- Retrieval-Augmented Generation (RAG)
- Local Large Language Models
- Semantic Search
- Vector Databases
- Clean Architecture
- FastAPI
- Streamlit

The goal is to build an AI application that is reliable, modular, maintainable, and suitable for real-world document processing workflows.

---

# Repository

```
SDS-Metadata
│
├── app/
├── frontend/
├── tests/
├── docs/
├── scripts/
├── data/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# Author

**Abhijit Kumar**

GitHub

https://github.com/ps-abhijit-kumar

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a star.

It helps others discover the project and supports future improvements.

---

**SDS Metadata Extraction System**

*Built with Python, FastAPI, Streamlit, Ollama, ChromaDB, SQLite, and Clean Architecture.*

</div>
