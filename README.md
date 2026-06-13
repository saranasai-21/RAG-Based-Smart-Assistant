# RAG-Based Smart Assistant

An advanced, production-minded **multi-document Retrieval-Augmented Generation
(RAG) assistant** built with Streamlit. Upload PDFs, Word documents, Markdown or
plain-text files and chat with them: ask questions, compare documents, get
summaries and trace every answer back to its source.

---

## Features

- **Multi-format ingestion** — `.pdf`, `.docx`, `.txt`, `.md`/`.markdown`.
- **Hybrid retrieval** — dense vector search (Chroma + BGE embeddings) fused with
  sparse **BM25** keyword search.
- **Cross-encoder reranking** — results are re-scored with a MiniLM cross-encoder
  for higher precision (lazily loaded — no model download just to import).
- **Multi-query expansion** — each question is rephrased into several semantic
  variants to improve recall.
- **Conversation memory** — follow-up questions are rewritten into standalone
  queries and recent turns are fed back into the prompt.
- **Document comparison mode** — holistic, document-wide comparison rather than
  isolated chunks.
- **Smart routing** — greetings, metadata questions ("how many docs?"),
  comparisons and normal QA are detected and handled differently.
- **Grounded answers with citations** — responses are constrained to retrieved
  context and show their sources (document + page + score).
- **Chat export** — download the full conversation as JSON.
- **Configurable everything** — models, chunking, retrieval and limits are all
  driven by environment variables.

---

## Architecture

```
streamlit_app.py          # Presentation layer (Streamlit only)
rag/
├── config.py             # Environment-driven settings (dataclass)
├── logging_config.py     # Centralised logging
├── text_utils.py         # Hashing, cleaning, chunking
├── query_classification.py  # Intent detection (pure functions)
├── prompts.py            # Prompt templates
├── loaders.py            # Multi-format document loaders
├── retrieval.py          # Embeddings, Chroma, BM25, reranking (lazy models)
└── llm.py                # LLM client + multi-query/summary/rewrite helpers
```

Heavy ML models are **loaded lazily and cached**, so importing the package never
triggers a download. This keeps start-up fast.

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier works)

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=...
```

Alternatively set the key in your shell or in `.streamlit/secrets.toml`.

### 4. Run

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501, upload documents in the sidebar and start chatting.

---

## Deploy to Render

The project includes a `render.yaml` and `Dockerfile` for one-click deployment.

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo.
3. Render will auto-detect the `render.yaml` and configure the service.
4. Add your `GROQ_API_KEY` as an environment variable in the Render dashboard.
5. Click **Deploy** — Render builds the Docker image and serves the app.

The service listens on the `PORT` environment variable (Render injects this automatically).

---

## Docker

```bash
docker build -t rag-assistant .
docker run -p 10000:10000 -e GROQ_API_KEY=your_key rag-assistant
```

---

## Configuration

All settings have sensible defaults and can be overridden via environment
variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | — | **Required.** Groq API key for the chat LLM. |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-embedding model. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Chat model served by Groq. |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `180` | Chunking parameters. |
| `MAX_FILE_SIZE_MB` | `25` | Per-file upload limit. |
| `MAX_TOTAL_CHUNKS` | `300` | Cap on indexed chunks. |
| `DEFAULT_TOP_K` | `8` | Chunks retrieved per query. |
| `DEFAULT_THRESHOLD` | `0.25` | Minimum similarity to keep a result. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store directory. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## License

This project is provided as-is for educational and internal use.
