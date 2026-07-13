# Quick Start Guide

## ✓ Everything is Already Running!

Both the **FastAPI backend** and **Streamlit frontend** are active and ready to use.

---

## 🚀 Open the Application

### **Open Your Browser:**
```
http://127.0.0.1:8501
```

That's it! The Streamlit UI will appear.

---

## 📝 How to Extract Metadata

### 1️⃣ Upload an SDS PDF
- Click **"Choose a file"** in the left sidebar
- Select a Safety Data Sheet (SDS) PDF document
- The file uploads automatically

### 2️⃣ Extract Metadata
- Click the **"Extract Metadata"** button
- Watch the extraction pipeline progress in real-time

### 3️⃣ View Results
The application will display:
- **File ID** – Unique document identifier
- **Product Name** – Chemical/material product name  
- **Language** – Detected language (English, Portuguese, Spanish, etc.)
- **Jurisdiction** – Detected regulatory framework (OSHA, REACH, ABNT, etc.)

### 4️⃣ View Document History
- Check **"View Extracted Documents"** tab to see all processed files
- Delete documents if needed
- Re-extract metadata anytime

---

## 🌐 API Access (Advanced Users)

### View API Documentation
```
http://127.0.0.1:8000/docs
```

### Extract via API
```bash
curl -X POST http://127.0.0.1:8000/api/v1/extract \
  -F "file=@sample.pdf"
```

### List Extracted Documents
```bash
curl http://127.0.0.1:8000/api/v1/documents
```

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

---

## ✅ Verify Everything is Working

### Check Backend Status
```bash
cd c:\Coding\Projects\SDS-Metadata-Extractor
.venv\Scripts\python.exe -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); print(r.json())"
```

### Check Frontend Status
Open: http://127.0.0.1:8501

---

## 🛠️ If Something Stops Working

### Restart the Backend
```bash
# Stop the process and restart
# The backend auto-reloads when code changes
```

### Restart the Frontend
```bash
# Streamlit auto-reloads when code changes
# Refresh your browser (F5)
```

### Reset Everything
```bash
cd c:\Coding\Projects\SDS-Metadata-Extractor
.venv\Scripts\python.exe scripts/init_db.py
# Then refresh your browser
```

---

## 📊 What Happens Behind the Scenes

When you upload an SDS PDF, the system:

1. ✓ Validates the PDF file
2. ✓ Extracts all text from all pages
3. ✓ Cleans and normalizes the text
4. ✓ Detects standard SDS sections (16 patterns recognized)
5. ✓ Splits into semantic chunks (1024 chars, 256 overlap)
6. ✓ Generates embeddings using **nomic-embed-text**
7. ✓ Stores vectors in **ChromaDB**
8. ✓ Performs semantic search to find relevant chunks
9. ✓ Sends chunks + prompt to **Ollama Qwen3:8B** LLM
10. ✓ Extracts metadata (File ID, Product Name, Language, Jurisdiction)
11. ✓ Validates results
12. ✓ Stores in **SQLite** database
13. ✓ Returns results to UI

**All processing happens locally. No cloud APIs. No internet required.**

---

## 📋 Supported File Types

- ✅ PDF documents (`.pdf`)
- ✅ Safety Data Sheets (SDS) in any format
- ✅ English, Portuguese, Spanish, and other languages
- ✅ Any regulatory framework (OSHA, REACH, ABNT NBR 14725, etc.)

---

## 🔧 Configuration

### Change Upload Directory
Edit `.env`:
```
UPLOAD_DIR=data/uploads
```

### Change Extraction Settings
Edit `app/infrastructure/configuration/settings.py`:
```python
CHUNK_SIZE = 1024        # Character size per chunk
CHUNK_OVERLAP = 256      # Character overlap between chunks
RETRIEVAL_K = 5          # Number of chunks to retrieve
```

### Change LLM Model
Edit `.env`:
```
OLLAMA_LLM_MODEL=mistral:7b
OLLAMA_EMBEDDING_MODEL=mistral-embed
```

---

## 📱 Technology Used

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.115.9 |
| **Frontend** | Streamlit | Latest |
| **LLM** | Ollama + Qwen3:8B | Latest |
| **Embeddings** | nomic-embed-text | Latest |
| **Vector DB** | ChromaDB | 1.5.9 |
| **Metadata DB** | SQLite | 3.x |
| **PDF Parser** | PyMuPDF | Latest |
| **Orchestration** | LangChain | 0.3.25 |
| **Language** | Python | 3.12+ |

---

## 💡 Tips & Tricks

### Tip 1: Multiple Documents
Upload multiple PDFs one by one. Each is processed independently.

### Tip 2: Monitor Progress
Check the logs in real-time:
```bash
tail -f logs/app.log
```

### Tip 3: API Integration
Use the REST API to integrate with other systems:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/extract \
  -F "file=@document.pdf" \
  | jq .metadata
```

### Tip 4: Clear History
Delete extracted documents from the UI to clean up:
- View Extracted Documents tab
- Click delete on any document

### Tip 5: Batch Processing
To process multiple files programmatically:
```python
import httpx

files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
client = httpx.Client(base_url='http://127.0.0.1:8000')

for file in files:
    with open(file, 'rb') as f:
        response = client.post(
            '/api/v1/extract',
            files={'file': f}
        )
        print(response.json()['metadata'])
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Connection refused"** | Backend not running. Check port 8000. |
| **"Ollama unreachable"** | Ensure `ollama serve` is running on port 11434. |
| **"Database locked"** | Restart backend. SQLite WAL mode will recover. |
| **"Upload fails"** | Ensure PDF is valid. Check file size. |
| **"Metadata incorrect"** | Different SDS formats may vary. Try another document. |
| **"Slow extraction"** | Normal for first run. Subsequent extractions are cached. |

---

## 📞 Support

- **API Docs:** http://127.0.0.1:8000/docs
- **Logs:** `logs/app.log`
- **Database:** `data/platform.db`
- **Vector Store:** `data/chroma_db`

---

## 🎯 Next Steps

✅ Open http://127.0.0.1:8501  
✅ Upload your first SDS PDF  
✅ Extract metadata  
✅ Verify results  
✅ Try batch processing  
✅ Explore the API  

**You're ready to go!**
