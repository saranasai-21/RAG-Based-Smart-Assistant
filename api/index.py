import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CHROMA_PERSIST_DIR"] = "/tmp/chroma_db"
os.environ["HF_HOME"] = "/tmp/hf"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import logging

logger = logging.getLogger(__name__)

from rag.config import get_settings
from rag.loaders import load_document
from rag.text_utils import chunk_text
from rag.llm import (
    load_llm, MissingAPIKeyError, summarize_document, 
    rewrite_followup_query, detect_relevant_documents, generate_multi_queries
)
from rag.query_classification import is_followup_query
from rag.prompts import build_qa_prompt
from rag.retrieval import (
    create_bm25, hybrid_search, 
    rerank_results, filter_results_by_documents
)

app = FastAPI(title="DocMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory application state (ephemeral per container restart)
GLOBAL_STATE = {
    "vector_db": None,
    "bm25": None,
    "chunks": [],
    "metadata": [],
    "document_summaries": {},
    "uploaded_hashes": set(),
    "uploaded_file_names": [],
    "chat_history": []
}

@app.get("/api/health")
def health():
    return {
        "status": "ok", 
        "docs": len(GLOBAL_STATE["uploaded_file_names"]),
        "file_names": GLOBAL_STATE["uploaded_file_names"]
    }

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    settings = get_settings()
    
    api_key = os.getenv("GROQ_API_KEY", "")
    try:
        llm = load_llm(api_key=api_key)
    except MissingAPIKeyError:
        raise HTTPException(status_code=401, detail="GROQ_API_KEY is missing.")

    all_chunks = GLOBAL_STATE["chunks"]
    all_metadata = GLOBAL_STATE["metadata"]
    new_file_names = list(GLOBAL_STATE["uploaded_file_names"])
    
    for uploaded_file in files:
        content = await uploaded_file.read()
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(status_code=400, detail=f"{uploaded_file.filename} exceeds size limit.")
            
        suffix = os.path.splitext(uploaded_file.filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
            
        try:
            pages = load_document(temp_path)
            doc_text = ""
            for page_num, text in pages:
                doc_text += text + "\n"
                chunks = chunk_text(text)[:settings.max_chunks_per_page]
                for idx, chunk in enumerate(chunks):
                    tagged_chunk = f"[SOURCE: {uploaded_file.filename}]\n[PAGE: {page_num}]\n\n{chunk}"
                    all_chunks.append(tagged_chunk)
                    all_metadata.append({"source": uploaded_file.filename, "page": page_num, "chunk_id": idx})
            
            summary = summarize_document(llm, doc_text, uploaded_file.filename)
            GLOBAL_STATE["document_summaries"][uploaded_file.filename] = summary
            if uploaded_file.filename not in new_file_names:
                new_file_names.append(uploaded_file.filename)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No extractable text found.")

    bm25 = create_bm25(all_chunks)
    
    if not GLOBAL_STATE["vector_db"]:
        from rag.retrieval import init_vector_db
        GLOBAL_STATE["vector_db"] = init_vector_db()
        
    # We add all chunks for simplicity in this serverless state simulation
    # Ideally, we'd only add new ones, but ephemeral client needs full state
    try:
        GLOBAL_STATE["vector_db"].add_texts(texts=all_chunks, metadatas=all_metadata)
    except Exception as e:
        logger.warning(f"Failed to add to Chroma: {e}")

    GLOBAL_STATE["bm25"] = bm25
    GLOBAL_STATE["chunks"] = all_chunks
    GLOBAL_STATE["metadata"] = all_metadata
    GLOBAL_STATE["uploaded_file_names"] = new_file_names
    
    return {"message": "Files indexed", "count": len(new_file_names), "chunks": len(all_chunks)}

@app.post("/api/chat")
async def chat(request: dict):
    query = request.get("query")
    history = request.get("history", [])
    
    # Store query in server-side chat history (will be paired with response later)
    GLOBAL_STATE["_pending_query"] = query
    
    if not GLOBAL_STATE["bm25"]:
        raise HTTPException(status_code=400, detail="Please upload documents first.")
        
    api_key = os.getenv("GROQ_API_KEY", "")
    try:
        llm = load_llm(api_key=api_key)
    except MissingAPIKeyError:
        raise HTTPException(status_code=401, detail="GROQ_API_KEY is missing.")

    settings = get_settings()
    
    if is_followup_query(query):
        query = rewrite_followup_query(llm, query, history)
        
    relevant_docs = detect_relevant_documents(query, GLOBAL_STATE["document_summaries"], llm)
    queries = generate_multi_queries(llm, query)
    queries.append(query)
    
    all_results = []
    for q in set(queries):
        res = hybrid_search(q, GLOBAL_STATE["vector_db"], GLOBAL_STATE["bm25"], GLOBAL_STATE["chunks"], metadata=GLOBAL_STATE["metadata"])
        all_results.extend(res)
        
    filtered = filter_results_by_documents(all_results, relevant_docs)
    
    seen = set()
    unique_res = []
    for r in filtered:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_res.append(r)
            
    ranked = rerank_results(query, unique_res)[:settings.default_top_k]
    
    context = "\n\n---\n\n".join([r["text"] for r in ranked])
    
    from rag.llm import get_recent_chat_history
    from rag.retrieval import format_sources
    sources_text = format_sources(ranked)
    history_text = get_recent_chat_history(history)
    
    from rag.agent import agent_app
    
    async def generate():
        import json
        full_generation = ""
        yield f"data: {json.dumps({'type': 'sources', 'data': list(set([r.get('metadata', {}).get('source', '?') for r in ranked]))})}\n\n"
        
        try:
            # Stream events from LangGraph
            for event in agent_app.stream(
                {
                    "query": query, 
                    "context": context, 
                    "sources_text": sources_text, 
                    "history_text": history_text,
                    "iterations": 0
                }
            ):
                for node_name, state_update in event.items():
                    # Provide thinking/agentic reasoning updates to UI
                    yield f"data: {json.dumps({'type': 'agent_step', 'data': f'Agent executed: {node_name}'})}\n\n"
                    
                    if "generation" in state_update and node_name == "generate":
                        full_generation = state_update['generation']
                        yield f"data: {json.dumps({'type': 'token', 'data': state_update['generation']})}\n\n"
                        
                    if "confidence_score" in state_update:
                        yield f"data: {json.dumps({'type': 'confidence', 'data': state_update['confidence_score']})}\n\n"
                        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        
        # Save chat exchange to server-side history for PDF export
        if full_generation:
            GLOBAL_STATE["chat_history"].append({
                "user": GLOBAL_STATE.get("_pending_query", query),
                "assistant": full_generation
            })
            # Keep last 20 exchanges
            if len(GLOBAL_STATE["chat_history"]) > 20:
                GLOBAL_STATE["chat_history"] = GLOBAL_STATE["chat_history"][-20:]
            
        yield "data: [DONE]\n\n"
        
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/report")
async def generate_report(request: dict):
    from fpdf import FPDF
    import base64
    
    class PDFReport(FPDF):
        def header(self):
            # Brand Header
            self.set_font("Helvetica", style="B", size=15)
            self.set_text_color(59, 130, 246)  # Primary Blue
            self.cell(0, 12, txt="DocMind AI - Executive Intelligence Report", ln=1, align='L')
            # Accent Line
            self.set_fill_color(139, 92, 246)  # Purple
            self.rect(15, 23, 180, 1, 'F')
            self.ln(6)
            
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", style="I", size=8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, f"Page {self.page_no()} | Generated by DocMind AI", align='C')

    pdf = PDFReport()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    query = request.get("query", "General Summary")
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(40, 8, txt="Report Focus Query: ", ln=0)
    pdf.set_font("Helvetica", style="I", size=10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, txt=query, ln=1)
    pdf.ln(4)
    
    if not GLOBAL_STATE["document_summaries"]:
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(220, 38, 38)
        pdf.multi_cell(0, 8, txt="No documents uploaded yet. Please upload documents in the DocMind AI Knowledge Base to generate a summary report.")
    else:
        for doc, summary in GLOBAL_STATE["document_summaries"].items():
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.set_text_color(59, 130, 246)
            pdf.cell(0, 10, txt=f"Document: {doc}", ln=1)
            
            # Clean markdown bold tags and remove emojis
            import re
            clean_summary = summary.replace("**", "")
            clean_summary = re.sub(r'[^\x00-\x7F]+', '', clean_summary)
            
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(30, 41, 59)
            # Encode/decode to ensure latin-1 support
            clean_summary_encoded = clean_summary.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5.5, txt=clean_summary_encoded)
            pdf.ln(6)
            
    # Use server-side chat history (primary), merge with any frontend-sent history
    import re
    history = GLOBAL_STATE.get("chat_history", [])
    frontend_history = request.get("history", [])
    # Add any frontend history entries not already on the server
    existing_queries = set(h.get("user", "") for h in history)
    for fh in frontend_history:
        if fh.get("user", "") not in existing_queries:
            history.append(fh)
    if history:
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 10, txt="Chat History", ln=1)
        pdf.ln(2)
        
        for msg in history:
            # User Question
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(59, 130, 246)
            user_text = msg.get("user", "")
            user_text = re.sub(r'[^\x00-\x7F]+', '', user_text).encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, txt=f"Q: {user_text}")
            
            # Assistant Answer
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(71, 85, 105)
            asst_text = msg.get("assistant", "")
            asst_text = asst_text.replace("**", "")
            asst_text = re.sub(r'[^\x00-\x7F]+', '', asst_text).encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5.5, txt=f"A: {asst_text}")
            pdf.ln(4)

    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    elif isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
        
    return {"pdf_base64": base64.b64encode(pdf_bytes).decode('utf-8')}

# Mount static files for serving the frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")

if os.path.exists(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(public_dir, "index.html"))

    @app.get("/{file_path:path}")
    async def serve_public_files(file_path: str):
        full_path = os.path.join(public_dir, file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return FileResponse(full_path)
        return FileResponse(os.path.join(public_dir, "index.html"))
