import os
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
    create_vector_db, create_bm25, hybrid_search, 
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

    vector_db = create_vector_db(all_chunks, all_metadata)
    bm25 = create_bm25(all_chunks)

    GLOBAL_STATE["vector_db"] = vector_db
    GLOBAL_STATE["bm25"] = bm25
    GLOBAL_STATE["chunks"] = all_chunks
    GLOBAL_STATE["metadata"] = all_metadata
    GLOBAL_STATE["uploaded_file_names"] = new_file_names
    
    return {"message": "Files indexed", "count": len(new_file_names), "chunks": len(all_chunks)}

@app.post("/api/chat")
async def chat(request: dict):
    query = request.get("query")
    history = request.get("history", [])
    
    if not GLOBAL_STATE["vector_db"]:
        raise HTTPException(status_code=400, detail="Please upload documents first.")
        
    api_key = os.getenv("GROQ_API_KEY", "")
    try:
        llm = load_llm(api_key=api_key)
    except MissingAPIKeyError:
        raise HTTPException(status_code=401, detail="GROQ_API_KEY is missing.")

    settings = get_settings()
    
    if is_followup_query(query):
        query = rewrite_followup_query(llm, query, history)
        
    relevant_docs = detect_relevant_documents(llm, query, GLOBAL_STATE["document_summaries"])
    queries = generate_multi_queries(llm, query)
    queries.append(query)
    
    all_results = []
    for q in set(queries):
        res = hybrid_search(q, GLOBAL_STATE["vector_db"], GLOBAL_STATE["bm25"], GLOBAL_STATE["chunks"])
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
    prompt = build_qa_prompt(query, context, history)
    
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    
    sources = []
    for r in ranked:
        md = r.get("metadata", {})
        s = md.get("source", "Unknown")
        p = md.get("page", "?")
        sources.append(f"{s} (Page {p})")
        
    return {
        "response": response.content,
        "sources": list(set(sources))
    }
