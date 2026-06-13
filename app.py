import streamlit as st
import tempfile
import os

from rag.config import get_settings
from rag.loaders import load_document
from rag.text_utils import chunk_text
from rag.llm import load_llm, summarize_document
from rag.retrieval import create_vector_db, create_bm25
from rag.graph import build_advanced_rag_graph

st.set_page_config(page_title="DocMind AI - Advanced RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #080c18; color: #e2e8f0; }
.stSidebar { background-color: #0f1320; }
</style>
""", unsafe_allow_html=True)

if "chunks" not in st.session_state:
    st.session_state.update({
        "chunks": [],
        "metadata_list": [],
        "vector_db": None,
        "bm25": None,
        "document_summaries": {},
        "history": [],
        "graph": build_advanced_rag_graph()
    })

with st.sidebar:
    st.header("📁 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX, MD, TXT)", accept_multiple_files=True)
    
    if st.button("Index Documents") and uploaded_files:
        with st.spinner("Indexing documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state["document_summaries"]:
                    continue
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name
                    
                pages = load_document(temp_path)
                doc_text = ""
                for page_num, text in pages:
                    doc_text += text + "\n"
                    chunks = chunk_text(text)
                    for idx, chunk in enumerate(chunks):
                        st.session_state["chunks"].append(chunk)
                        st.session_state["metadata_list"].append({"source": uploaded_file.name, "page": page_num})
                
                os.remove(temp_path)
                
                llm = load_llm()
                summary = summarize_document(llm, doc_text, uploaded_file.name)
                st.session_state["document_summaries"][uploaded_file.name] = summary
                
            if st.session_state["chunks"]:
                st.session_state["vector_db"] = create_vector_db(st.session_state["chunks"], st.session_state["metadata_list"])
                st.session_state["bm25"] = create_bm25(st.session_state["chunks"])
            st.success("Indexed successfully!")

    st.divider()
    st.header("✨ Advanced Features")
    search_mode = st.radio("Search Strategy", ["Hybrid (RRF)", "Dense Only", "Sparse Only"])
    top_k = st.slider("Context Compression (Top K)", 3, 20, 10)
    
st.title("🧠 DocMind AI")
st.caption("Multi-Document Intelligence · Hybrid Search · Reranking")

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if prompt := st.chat_input("Ask anything about your documents..."):
    if not st.session_state["vector_db"]:
        st.error("Please upload and index documents first.")
    else:
        st.session_state["history"].append({"role": "user", "content": prompt, "user": prompt, "assistant": ""})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing and retrieving..."):
                state = {
                    "query": prompt,
                    "history": st.session_state["history"][:-1],
                    "vector_db": st.session_state["vector_db"],
                    "bm25": st.session_state["bm25"],
                    "chunks": st.session_state["chunks"],
                    "metadata_list": st.session_state["metadata_list"],
                    "document_summaries": st.session_state["document_summaries"]
                }
                
                result_state = st.session_state["graph"].invoke(state)
                
                st.markdown(result_state["response"])
                if result_state["sources"]:
                    with st.expander("Sources"):
                        for s in result_state["sources"]:
                            st.markdown(f"- {s}")
                            
                with st.expander("Advanced RAG Tracing"):
                    st.write("**Expanded Queries:**", result_state["expanded_queries"])
                    st.write("**Relevant Docs Detected:**", result_state["relevant_docs"])
                    st.write("**Reranked Chunks:**")
                    for c in result_state["reranked_chunks"]:
                        st.write(f"Score: {c.get('rerank_score', c.get('score', 0)):.4f} - {c['metadata']}")
                        
                st.session_state["history"][-1]["assistant"] = result_state["response"]
                st.session_state["history"][-1]["sources"] = result_state["sources"]
