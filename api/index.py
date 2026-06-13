from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import json

from rag.loaders import load_document
from rag.text_utils import chunk_text
from rag.llm import load_llm, summarize_document
from rag.retrieval import create_vector_db, create_bm25
from rag.graph import create_rag_graph

app = FastAPI(title="Advanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory global state for serverless
GLOBAL_STATE = {
    "chunks": [],
    "metadata": [],
    "vector_db": None,
    "bm25": None,
    "document_summaries": {}
}

rag_graph = create_rag_graph()

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
            
        pages = load_document(temp_path)
        doc_text = ""
        new_chunks = []
        new_metadata = []
        for page_num, text in pages:
            doc_text += text + "\n"
            chunks = chunk_text(text)
            for chunk in chunks:
                new_chunks.append(chunk)
                new_metadata.append({"source": file.filename, "page": page_num})
                
        os.remove(temp_path)
        
        GLOBAL_STATE["chunks"].extend(new_chunks)
        GLOBAL_STATE["metadata"].extend(new_metadata)
        
        llm = load_llm()
        summary = summarize_document(llm, doc_text, file.filename)
        GLOBAL_STATE["document_summaries"][file.filename] = summary
        
        GLOBAL_STATE["vector_db"] = create_vector_db(GLOBAL_STATE["chunks"], GLOBAL_STATE["metadata"])
        GLOBAL_STATE["bm25"] = create_bm25(GLOBAL_STATE["chunks"])
        
        return {"status": "success", "message": f"Indexed {len(new_chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not GLOBAL_STATE["chunks"]:
        raise HTTPException(status_code=400, detail="No documents indexed.")
        
    state = {
        "question": req.query,
        "chat_history": req.history,
        "global_state": GLOBAL_STATE,
        "intent": "",
        "expanded_queries": [],
        "retrieved_chunks": [],
        "generation": "",
        "sources": []
    }
    
    try:
        result = rag_graph.invoke(state)
        
        return {
            "answer": result["generation"],
            "sources": result["sources"],
            "intent": result["intent"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
