---
title: Rag Smart Assistant
emoji: ⚡
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
---

<div align="center">

# 🧠 DocMind AI — RAG-Based Smart Assistant

**An enterprise-grade Retrieval-Augmented Generation (RAG) assistant for intelligent document Q&A.**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/saranasai/Rag-Smart-Assistant)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6F00)](https://langchain-ai.github.io/langgraph/)

</div>

---

## 📖 Overview

DocMind AI is a full-stack **RAG-based smart assistant** that lets you upload documents (PDF, DOCX, TXT, Markdown) and ask natural-language questions grounded in your uploaded content. It combines **hybrid retrieval** (BM25 + ChromaDB semantic search), **multi-LLM generation** (Groq + Gemini), and an **agentic LangGraph pipeline** with hallucination checking — all served through a modern glassmorphism web UI.

### 🎬 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/saranasai/Rag-Smart-Assistant)**

<img width="1918" height="811" alt="image" src="https://github.com/user-attachments/assets/355d5e77-9e3c-4244-a947-a7dc3e6b0d1a" />

<img width="1530" height="696" alt="image" src="https://github.com/user-attachments/assets/0d742c1a-7cda-4ad4-947b-f952f08edaff" />

---

## ✨ Key Features

### 🔄 Complete RAG Pipeline

| Stage | Description | Technology |
|-------|-------------|------------|
| **📥 Document Ingestion** | Upload PDF, DOCX, TXT, and Markdown files | PyPDFLoader, python-docx |
| **✂️ Text Chunking** | Recursive character splitting with configurable overlap | LangChain `RecursiveCharacterTextSplitter` (900 chars / 180 overlap) |
| **🧮 Vector Embedding** | Dense embeddings stored in an in-memory vector store | ChromaDB + Google `embedding-001` model |
| **🔍 Semantic Retrieval** | Hybrid search combining sparse and dense retrieval | BM25 (0.3 weight) + ChromaDB semantic (0.7 weight) |
| **💡 LLM-Augmented Generation** | Agentic pipeline with hallucination grading | LangGraph agent → Groq (Llama 3.1) + Gemini 2.5 Flash |

### 🤖 Agentic Architecture (LangGraph)

The assistant uses a **LangGraph state machine** with three nodes:

```
┌──────────┐     ┌──────────┐     ┌─────────────────────┐
│ Retrieve │ ──▶ │ Generate │ ──▶ │ Grade Hallucination │
└──────────┘     └──────────┘     └─────────────────────┘
                      ▲                      │
                      └──── (retry if bad) ◀─┘
```

- **Retrieve**: Logs the retrieval step (retrieval is performed upstream for streaming)
- **Generate**: Invokes the Multi-LLM Router to produce a grounded answer
- **Grade Hallucination**: Checks if the generation is grounded; retries if not (max 2 iterations)

### 🧠 Multi-LLM Router

The system intelligently routes across multiple LLMs with automatic fallback:

1. **Groq (Primary)** — Llama 3.1 8B Instant for fast draft generation
2. **Groq (Secondary)** — Backup Groq key for rate-limit resilience
3. **Gemini 2.5 Flash** — Refines Groq's output or serves as full fallback

### 🎨 Premium Web UI

- **Glassmorphism** dark theme with animated gradient background orbs
- **Real-time streaming** responses with SSE (Server-Sent Events)
- **Typing indicator** (3-dot bounce) while the AI processes
- **RAG pipeline status pill** — animates through Ingest → Chunk → Embed → Retrieve → Generate
- **Source citations** and confidence scores on every answer
- **PDF report export** for document summaries
- **Analytics dashboard** with animated stats, bar chart, donut chart, and pipeline visualization
- **Fully mobile responsive** — hamburger menu, slide-over sidebar, adaptive layouts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (HTML/CSS/JS)               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Chat UI     │  │  Upload UI   │  │  Dashboard    │  │
│  │  (SSE stream)│  │  (drag/drop) │  │  (analytics)  │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────┘  │
│         │                 │                              │
└─────────┼─────────────────┼──────────────────────────────┘
          │ /api/chat       │ /api/upload
          ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                FastAPI Backend (api/index.py)             │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Loaders │→ │ Chunking │→ │ChromaDB  │→ │ Hybrid   │  │
│  │ PDF/DOCX│  │ Recursive│  │ + BM25   │  │ Search   │  │
│  └─────────┘  └──────────┘  └──────────┘  └────┬─────┘  │
│                                                 │        │
│  ┌──────────────────────────────────────────────▼─────┐  │
│  │            LangGraph Agent Pipeline                │  │
│  │  retrieve → generate → grade_hallucination → END   │  │
│  │              (Multi-LLM: Groq + Gemini)            │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG-Based-Smart-Assistant/
├── api/
│   └── index.py              # FastAPI server — all API endpoints
├── rag/
│   ├── __init__.py            # Package init
│   ├── agent.py               # LangGraph agentic pipeline
│   ├── config.py              # Environment-driven settings
│   ├── llm.py                 # Multi-LLM Router (Groq + Gemini)
│   ├── loaders.py             # PDF, DOCX, TXT, MD document loaders
│   ├── logging_config.py      # Centralized logging setup
│   ├── prompts.py             # All prompt templates
│   ├── query_classification.py# Intent detection (comparison, followup, etc.)
│   ├── retrieval.py           # ChromaDB + BM25 hybrid search
│   └── text_utils.py          # Text cleaning and chunking
├── public/
│   ├── index.html             # Main chat interface
│   └── dashboard.html         # Analytics dashboard
├── Dockerfile                 # Docker config for HF Spaces
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata & tooling config
├── start.sh                   # Startup script
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for:
  - [Groq](https://console.groq.com/) — LLM inference
  - [Google AI / Gemini](https://aistudio.google.com/apikey) — embeddings + LLM fallback

### Local Development

1. **Clone the repository:**

   ```bash
   git clone https://github.com/saranasai-21/RAG-Based-Smart-Assistant.git
   cd RAG-Based-Smart-Assistant
   ```

2. **Create a `.env` file:**

   ```env
   GROQ_API_KEY_1=gsk_your_groq_key_here
   GROQ_API_KEY_2=gsk_your_backup_key_here   # optional
   GEMINI_API_KEY=your_gemini_key_here
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**

   ```bash
   uvicorn api.index:app --host 0.0.0.0 --port 7860
   ```

5. **Open in browser:** [http://localhost:7860](http://localhost:7860)

### Docker

```bash
docker build -t docmind-ai .
docker run -p 7860:7860 --env-file .env docmind-ai
```

---

## ☁️ Deployment

### Hugging Face Spaces (Recommended)

This project is configured for **HF Spaces with Docker SDK**:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repo to the Space
4. Set your API keys as **Secrets** in the Space settings:
   - `GROQ_API_KEY_1`
   - `GROQ_API_KEY_2` (optional)
   - `GEMINI_API_KEY`

---

## ⚙️ Configuration

All settings are environment-driven via [`rag/config.py`](rag/config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `CHUNK_SIZE` | `900` | Characters per text chunk |
| `CHUNK_OVERLAP` | `180` | Overlap between chunks |
| `MIN_CHUNK_LEN` | `100` | Minimum chunk length to keep |
| `DEFAULT_TOP_K` | `8` | Number of retrieval results |
| `DEFAULT_THRESHOLD` | `0.25` | Minimum relevance score |
| `MAX_FILE_SIZE_MB` | `100` | Max upload file size |
| `SUMMARY_CHAR_LIMIT` | `15000` | Max chars for document summarization |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Vanilla HTML/CSS/JS, Glassmorphism UI, Outfit font |
| **Backend** | FastAPI, Uvicorn |
| **LLM Orchestration** | LangChain, LangGraph |
| **LLMs** | Groq (Llama 3.1 8B), Google Gemini 2.5 Flash |
| **Embeddings** | Google Generative AI `embedding-001` |
| **Vector Store** | ChromaDB (in-memory) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Document Parsing** | PyPDFLoader, python-docx |
| **PDF Reports** | fpdf2 |
| **Deployment** | Docker, Hugging Face Spaces |

---

## 📄 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check — returns doc count |
| `POST` | `/api/upload` | Upload documents (multipart form) |
| `POST` | `/api/chat` | Chat query (SSE streaming response) |
| `POST` | `/api/report` | Generate PDF report (base64) |
| `GET` | `/` | Serve main chat UI |
| `GET` | `/dashboard.html` | Serve analytics dashboard |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


