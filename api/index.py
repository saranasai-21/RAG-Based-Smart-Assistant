import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CHROMA_PERSIST_DIR"] = "/tmp/chroma_db"
os.environ["HF_HOME"] = "/tmp/hf"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import uuid

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

# Vercel Serverless state (Ephemeral)
GLOBAL_STATE = {
    "vector_db": None,
    "bm25": None,
    "chunks": [],
    "metadata": [],
    "document_summaries": {},
    "uploaded_hashes": set(),
    "uploaded_file_names": []
}

@app.get("/api/health")
def health():
    return {"status": "ok", "docs": len(GLOBAL_STATE["uploaded_file_names"])}

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
                        yield f"data: {json.dumps({'type': 'token', 'data': state_update['generation']})}\n\n"
                        
                    if "confidence_score" in state_update:
                        yield f"data: {json.dumps({'type': 'confidence', 'data': state_update['confidence_score']})}\n\n"
                        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/report")
async def generate_report(request: dict):
    from fpdf import FPDF
    import base64
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="RAG Smart Assistant - Analytics & Report", ln=1, align='C')
    
    query = request.get("query", "General Summary")
    pdf.cell(200, 10, txt=f"Query Focus: {query}", ln=1)
    
    summary_text = ""
    for doc, summary in GLOBAL_STATE["document_summaries"].items():
        summary_text += f"\nDocument: {doc}\nSummary: {summary}\n"
        
    pdf.multi_cell(0, 10, txt=summary_text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return {"pdf_base64": base64.b64encode(pdf_bytes).decode('utf-8')}
