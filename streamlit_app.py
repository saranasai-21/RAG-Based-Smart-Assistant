"""Streamlit UI for the RAG-Based Smart Assistant."""

import json
import os
import tempfile
import time

import streamlit as st

from rag.config import get_settings
from rag.llm import (
    MissingAPIKeyError,
    detect_relevant_documents,
    generate_multi_queries,
    get_recent_chat_history,
    load_llm,
    rewrite_followup_query,
    summarize_document,
)
from rag.loaders import SUPPORTED_EXTENSIONS, load_document
from rag.prompts import build_comparison_prompt, build_qa_prompt
from rag.query_classification import (
    is_comparison_query,
    is_followup_query,
    is_general_chat,
    is_metadata_query,
)
from rag.retrieval import (
    create_bm25,
    create_vector_db,
    filter_results_by_documents,
    format_sources,
    hybrid_search,
    rerank_results,
)
from rag.text_utils import chunk_text, get_file_hash

settings = get_settings()

UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]

FILE_ICONS = {
    "pdf": "📕",
    "docx": "📘",
    "doc": "📘",
    "txt": "📄",
    "md": "📝",
    "markdown": "📝",
}

SEARCH_MODE_CONFIG = {
    "⚡ Fast":        {"top_k": 4,  "threshold": 0.40},
    "⚖️ Balanced":   {"top_k": 8,  "threshold": 0.25},
    "📚 Deep Search": {"top_k": 15, "threshold": 0.15},
}


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
/* ── Fonts ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ─────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #080c18 0%, #0d1229 60%, #080c18 100%);
    min-height: 100vh;
}

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1320 0%, #0b0f1c 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.18);
    box-shadow: 4px 0 32px rgba(99,102,241,0.07);
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem 2rem; }

/* Section headers */
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4b5563;
    margin: 1.2rem 0 0.5rem;
    padding: 0 2px;
}

/* ── Glassmorphism card ─────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 14px 14px 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
}
.glass-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6366f1;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.05) !important;
    border: 1.5px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(99,102,241,0.08) !important;
}
[data-testid="stFileUploader"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stFileUploader"] small { color: #4b5563 !important; font-size: 0.72rem !important; }

/* ── File list items ────────────────────────────────── */
.file-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 8px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    margin-bottom: 5px;
    font-size: 0.78rem;
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.file-item-icon { font-size: 0.9rem; flex-shrink: 0; }
.file-item-name { overflow: hidden; text-overflow: ellipsis; flex: 1; }

/* ── Stats card ─────────────────────────────────────── */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 4px;
}
.stat-cell {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.13);
    border-radius: 10px;
    padding: 8px 10px;
    text-align: center;
}
.stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.65rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 1px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 7px;
    width: 100%;
    justify-content: center;
}
.status-ready   { background: rgba(16,185,129,0.12); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.25); }
.status-idle    { background: rgba(75,85,99,0.15);   color: #4b5563; border: 1px solid rgba(75,85,99,0.2);   }

/* ── Search mode radio ──────────────────────────────── */
[data-testid="stRadio"] label { color: #94a3b8 !important; font-size: 0.82rem !important; }
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
    flex-direction: column !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 9px !important;
    padding: 7px 10px !important;
    transition: all 0.2s;
    width: 100%;
    cursor: pointer;
}
[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    color: #a5b4fc !important;
}

/* ── Memory toggle ──────────────────────────────────── */
[data-testid="stToggle"] label { color: #94a3b8 !important; font-size: 0.82rem !important; }
[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
    background-color: #6366f1 !important;
}

/* ── Checkbox (advanced) ────────────────────────────── */
[data-testid="stCheckbox"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* ── Slider (advanced) ──────────────────────────────── */
[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #6366f1 !important;
    border: 2px solid #818cf8 !important;
    box-shadow: 0 0 8px rgba(99,102,241,0.5) !important;
}

/* ── Sidebar buttons ────────────────────────────────── */
[data-testid="stSidebar"] .stButton button {
    background: rgba(239,68,68,0.08) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    width: 100%;
    padding: 6px 12px !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(239,68,68,0.15) !important;
    border-color: rgba(239,68,68,0.4) !important;
}

/* ── Sidebar expander ───────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(99,102,241,0.04) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #6366f1 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ── Divider ────────────────────────────────────────── */
[data-testid="stSidebar"] hr {
    border-color: rgba(99,102,241,0.12) !important;
    margin: 10px 0 !important;
}

/* ── Main content ───────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 860px;
}

/* ── Header ─────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 1.4rem 2rem 1.6rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(6,182,212,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 20px;
    margin-bottom: 1.6rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
}
.app-header p { color: #4b5563; font-size: 0.85rem; margin: 0; }

/* ── Chat messages ───────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg,rgba(99,102,241,0.13),rgba(79,70,229,0.08)) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(13,18,35,0.85) !important;
    border: 1px solid rgba(6,182,212,0.18) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span { color: #e2e8f0 !important; }

/* ── Chat toolbar row ───────────────────────────────── */
.chat-toolbar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 6px;
}

/* ── Chat input ─────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(13,18,35,0.9) !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.08) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 28px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] textarea { color: #e2e8f0 !important; background: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #374151 !important; }

/* ── Popover ─────────────────────────────────────────── */
[data-testid="stPopover"] > button {
    background: rgba(99,102,241,0.1) !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    padding: 5px 12px !important;
    transition: all 0.2s;
}
[data-testid="stPopover"] > button:hover {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.45) !important;
}
[data-baseweb="popover"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Expanders (main) ────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(13,18,35,0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #6b7280 !important; font-size: 0.83rem !important; font-weight: 500 !important; }
[data-testid="stExpander"] summary:hover { color: #818cf8 !important; }

/* ── Source cards ────────────────────────────────────── */
.source-card {
    background: rgba(13,18,35,0.9);
    border: 1px solid rgba(99,102,241,0.18);
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-left-color 0.2s;
}
.source-card:hover { border-left-color: #22d3ee; }
.badge-row { display: flex; gap: 8px; margin-bottom: 7px; flex-wrap: wrap; }
.badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-doc   { background: rgba(99,102,241,0.15); color: #a5b4fc; border:1px solid rgba(99,102,241,0.25);}
.badge-page  { background: rgba(6,182,212,0.12);  color: #38bdf8; border:1px solid rgba(6,182,212,0.22);}
.badge-score { background: rgba(52,211,153,0.1);  color: #6ee7b7; border:1px solid rgba(52,211,153,0.18);}
.source-preview {
    color: #4b5563;
    font-size: 0.76rem;
    border-top: 1px solid rgba(99,102,241,0.1);
    padding-top: 7px;
    margin-top: 5px;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Alerts ──────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 12px !important; border: none !important; }

/* ── Download button ─────────────────────────────────── */
.stDownloadButton button {
    background: linear-gradient(135deg, #4f46e5, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
    transition: all 0.2s;
}
.stDownloadButton button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(79,70,229,0.4) !important; }

/* ── Welcome card ────────────────────────────────────── */
.welcome-card { text-align: center; padding: 3rem 2rem; }
.welcome-card .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.welcome-card h3 { color: #374151; font-size: 1.1rem; margin-bottom: 0.4rem; }
.welcome-card p  { color: #1f2937; font-size: 0.85rem; }

/* ── Misc ─────────────────────────────────────────────── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e2e8f0 !important; }
.stMarkdown p, .stMarkdown li { color: #94a3b8 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080c18; }
::-webkit-scrollbar-thumb { background: #1e1b4b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "messages": [],
    "chat_history": [],
    "processed": False,
    "vector_db": None,
    "bm25": None,
    "chunks": None,
    "metadata": None,
    "document_summaries": {},
    "uploaded_hashes": set(),
    "last_query": "",
    "uploaded_file_names": [],
    "search_mode": "⚖️ Balanced",
    "use_memory": True,
    "show_sources": True,
    "use_reranker": True,
    "adv_top_k": 8,
    "adv_threshold": 0.25,
    "use_advanced_override": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    # ── Section 1: Knowledge Base ──────────────────────
    st.markdown('<div class="sidebar-section-label">📁 Knowledge Base</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div style="font-size:0.7rem;color:#374151;text-align:center;margin-top:4px;">'
        f'Supports: {", ".join(t.upper() for t in UPLOAD_TYPES)}</div>',
        unsafe_allow_html=True,
    )

    # File list
    if st.session_state.uploaded_file_names:
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        for fname in st.session_state.uploaded_file_names:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            icon = FILE_ICONS.get(ext, "📄")
            st.markdown(
                f'<div class="file-item"><span class="file-item-icon">{icon}</span>'
                f'<span class="file-item-name">{fname}</span></div>',
                unsafe_allow_html=True,
            )
        doc_count = len(st.session_state.uploaded_file_names)
        st.markdown(
            f'<div style="font-size:0.72rem;color:#4b5563;margin-top:6px;text-align:right;">'
            f'Documents: {doc_count}</div>',
            unsafe_allow_html=True,
        )

    # ── Section 2: Search Mode ─────────────────────────
    st.markdown('<div class="sidebar-section-label">🔍 Search Mode</div>', unsafe_allow_html=True)
    search_mode = st.radio(
        "search_mode_radio",
        list(SEARCH_MODE_CONFIG.keys()),
        index=list(SEARCH_MODE_CONFIG.keys()).index(st.session_state.search_mode),
        label_visibility="collapsed",
    )
    st.session_state.search_mode = search_mode

    # ── Section 3: Memory ──────────────────────────────
    st.markdown('<div class="sidebar-section-label">🧠 Memory</div>', unsafe_allow_html=True)
    use_memory = st.toggle(
        "Enable Conversation Memory",
        value=st.session_state.use_memory,
    )
    st.session_state.use_memory = use_memory

    # ── Section 4: Statistics ──────────────────────────
    st.markdown('<div class="sidebar-section-label">📊 Statistics</div>', unsafe_allow_html=True)
    doc_count = len(st.session_state.uploaded_file_names)
    chunk_count = len(st.session_state.chunks) if st.session_state.chunks else 0
    is_ready = st.session_state.processed

    st.markdown(
        f"""
<div class="glass-card">
  <div class="stats-grid">
    <div class="stat-cell">
      <div class="stat-value">{doc_count}</div>
      <div class="stat-label">Documents</div>
    </div>
    <div class="stat-cell">
      <div class="stat-value">{chunk_count}</div>
      <div class="stat-label">Chunks</div>
    </div>
  </div>
  <div style="text-align:center;">
    {'<span class="status-pill status-ready">● Active</span>' if is_ready
     else '<span class="status-pill status-idle">○ Awaiting Documents</span>'}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Section 5: Advanced Settings ──────────────────
    with st.expander("⚙️ Advanced Settings"):
        use_reranker = st.checkbox("Enable Reranker", value=st.session_state.use_reranker)
        st.session_state.use_reranker = use_reranker

        show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
        st.session_state.show_sources = show_sources

        use_advanced_override = st.checkbox(
            "Override Search Mode manually",
            value=st.session_state.use_advanced_override,
        )
        st.session_state.use_advanced_override = use_advanced_override

        if use_advanced_override:
            adv_top_k = st.slider(
                "Retrieved Chunks", 2, 20, st.session_state.adv_top_k
            )
            adv_threshold = st.slider(
                "Similarity Threshold", 0.0, 1.0, st.session_state.adv_threshold, 0.05
            )
            st.session_state.adv_top_k = adv_top_k
            st.session_state.adv_threshold = adv_threshold

    st.markdown("---")
    clear_chat = st.button("🗑️ Clear Knowledge Base & Chat")


# ── Resolve top_k / threshold from search mode ────────
if st.session_state.use_advanced_override:
    top_k = st.session_state.adv_top_k
    threshold = st.session_state.adv_threshold
else:
    cfg = SEARCH_MODE_CONFIG[st.session_state.search_mode]
    top_k = cfg["top_k"]
    threshold = cfg["threshold"]


# =====================================================
# CLEAR STATE
# =====================================================

if clear_chat:
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class="app-header">
    <h1>🧠 DocMind AI</h1>
    <p>Multi-Document Intelligence &nbsp;·&nbsp; Hybrid Search &nbsp;·&nbsp; Reranking &nbsp;·&nbsp; Conversation Memory</p>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================
# LOAD LLM
# =====================================================


def _read_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "") or _read_secret("GROQ_API_KEY")
    return load_llm(api_key=api_key)


try:
    llm = get_llm()
except MissingAPIKeyError:
    st.error(
        "🔑 **GROQ_API_KEY is not configured.** Add it to your environment or "
        "`.streamlit/secrets.toml` to start chatting."
    )
    st.stop()


# =====================================================
# PROCESS DOCUMENTS
# =====================================================

if uploaded_files and not st.session_state.processed:
    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    document_summaries: dict[str, str] = {}
    new_file_names: list[str] = list(st.session_state.uploaded_file_names)

    for uploaded_file in uploaded_files:
        if uploaded_file.size > settings.max_file_size_bytes:
            st.error(f"**{uploaded_file.name}** exceeds the {settings.max_file_size_mb} MB limit.")
            st.stop()

        file_hash = get_file_hash(uploaded_file)
        if file_hash in st.session_state.uploaded_hashes:
            continue
        st.session_state.uploaded_hashes.add(file_hash)

        if uploaded_file.name not in new_file_names:
            new_file_names.append(uploaded_file.name)

        suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"
        with st.spinner(f"📄 Indexing **{uploaded_file.name}**…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            try:
                pages = load_document(temp_path)
            except Exception as exc:
                st.error(f"Failed to load **{uploaded_file.name}**: {exc}")
                st.stop()
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            doc_text = ""
            for page_num, text in pages:
                doc_text += text + "\n"
                chunks = chunk_text(text)[: settings.max_chunks_per_page]
                for idx, chunk in enumerate(chunks):
                    tagged_chunk = f"[SOURCE: {uploaded_file.name}]\n[PAGE: {page_num}]\n\n{chunk}"
                    all_chunks.append(tagged_chunk)
                    all_metadata.append(
                        {"source": uploaded_file.name, "page": page_num, "chunk_id": idx}
                    )

            summary = summarize_document(llm, doc_text, uploaded_file.name)
            document_summaries[uploaded_file.name] = summary

    all_chunks = all_chunks[: settings.max_total_chunks]
    all_metadata = all_metadata[: settings.max_total_chunks]

    if not all_chunks:
        st.warning("⚠️ No extractable text found in the uploaded documents.")
        st.stop()

    st.success(f"✅ Indexed **{len(all_chunks)}** chunks across **{len(new_file_names)}** document(s).")

    with st.spinner("🧠 Building vector index…"):
        vector_db = create_vector_db(all_chunks, all_metadata)

    bm25 = create_bm25(all_chunks)

    st.session_state.vector_db = vector_db
    st.session_state.bm25 = bm25
    st.session_state.chunks = all_chunks
    st.session_state.metadata = all_metadata
    st.session_state.document_summaries = document_summaries
    st.session_state.uploaded_file_names = new_file_names
    st.session_state.processed = True

    st.success("🚀 **Knowledge base ready.** Start asking questions below!")


# =====================================================
# CHAT HISTORY
# =====================================================

if not st.session_state.messages:
    st.markdown(
        """
<div class="welcome-card">
    <div class="icon">💬</div>
    <h3>Your AI document assistant is ready</h3>
    <p>Upload documents in the sidebar, then ask anything about them.</p>
</div>
""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# CHAT TOOLBAR — settings popover near input
# =====================================================

toolbar_col, input_col = st.columns([1, 11])

with toolbar_col:
    with st.popover("⚙️"):
        st.markdown("**Search Mode**")
        pop_mode = st.radio(
            "pop_search_mode",
            list(SEARCH_MODE_CONFIG.keys()),
            index=list(SEARCH_MODE_CONFIG.keys()).index(st.session_state.search_mode),
            label_visibility="collapsed",
        )
        if pop_mode != st.session_state.search_mode:
            st.session_state.search_mode = pop_mode
            st.rerun()

        st.markdown("---")
        pop_memory = st.toggle(
            "🧠 Conversation Memory",
            value=st.session_state.use_memory,
            key="pop_memory",
        )
        if pop_memory != st.session_state.use_memory:
            st.session_state.use_memory = pop_memory
            st.rerun()

        st.markdown("---")
        pop_sources = st.toggle(
            "📎 Show Sources",
            value=st.session_state.show_sources,
            key="pop_sources",
        )
        if pop_sources != st.session_state.show_sources:
            st.session_state.show_sources = pop_sources
            st.rerun()

with input_col:
    query = st.chat_input("Ask anything about your documents…")


# =====================================================
# CHAT LOGIC
# =====================================================

if query:
    if not st.session_state.processed:
        st.warning("⚠️ Please upload documents first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    response = ""
    original_query = query

    if is_followup_query(query):
        query = rewrite_followup_query(llm, query, st.session_state.chat_history)
        with st.expander("🧠 Rewritten Query"):
            st.write(query)

    if is_metadata_query(query):
        doc_names = list(st.session_state.document_summaries.keys())
        response = f"You have **{len(doc_names)}** document(s) in the knowledge base:\n\n"
        for idx, doc in enumerate(doc_names, start=1):
            ext = doc.rsplit(".", 1)[-1].lower() if "." in doc else ""
            icon = FILE_ICONS.get(ext, "📄")
            response += f"{idx}. {icon} {doc}\n"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    if is_general_chat(query):
        response = (
            "👋 **Hello!** I'm **DocMind AI**, your document intelligence assistant.\n\n"
            "I can help you:\n"
            "- 🔍 **Answer questions** from uploaded documents\n"
            "- ⚖️ **Compare** documents side-by-side\n"
            "- 📝 **Summarize** files on demand\n"
            "- 💡 **Extract key insights** from technical content\n"
            "- 🧠 **Remember** context across your conversation\n\n"
            "Upload documents in the sidebar and start asking!"
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    retrieved_chunks: list[dict] = []

    if is_comparison_query(query):
        comparison_prompt = build_comparison_prompt(query, st.session_state.document_summaries)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in llm.stream(comparison_prompt):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                response += content
                placeholder.markdown(response)
                time.sleep(0.01)
    else:
        relevant_docs = detect_relevant_documents(query, st.session_state.document_summaries, llm)
        with st.expander("📂 Relevant Documents Detected"):
            if relevant_docs:
                for doc in relevant_docs:
                    ext = doc.rsplit(".", 1)[-1].lower() if "." in doc else ""
                    icon = FILE_ICONS.get(ext, "📄")
                    st.markdown(f"- {icon} {doc}")
            else:
                st.write("No specific documents detected — searching across all.")

        retrieved_results = []
        for q in generate_multi_queries(llm, query):
            retrieved_results.extend(
                hybrid_search(
                    query=q,
                    vector_db=st.session_state.vector_db,
                    bm25=st.session_state.bm25,
                    chunks=st.session_state.chunks,
                    k=top_k,
                    threshold=threshold,
                )
            )

        retrieved_results = filter_results_by_documents(retrieved_results, relevant_docs)

        unique_results: list[dict] = []
        seen: set = set()
        for item in retrieved_results:
            if item["text"] not in seen:
                unique_results.append(item)
                seen.add(item["text"])

        if st.session_state.use_reranker:
            unique_results = rerank_results(query, unique_results)

        retrieved_chunks = unique_results[:top_k]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            if not retrieved_chunks:
                response = "⚠️ No relevant information found in the knowledge base."
                placeholder.markdown(response)
            else:
                context = "\n\n".join(item["text"] for item in retrieved_chunks)
                sources_text = format_sources(retrieved_chunks)
                history_text = (
                    get_recent_chat_history(st.session_state.chat_history)
                    if st.session_state.use_memory
                    else ""
                )
                prompt = build_qa_prompt(context, sources_text, query, history_text)
                for chunk in llm.stream(prompt):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    response += content
                    placeholder.markdown(response)
                    time.sleep(0.01)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append({"user": original_query, "assistant": response})

    with st.expander("📘 Document Summaries"):
        for doc, summary in st.session_state.document_summaries.items():
            ext = doc.rsplit(".", 1)[-1].lower() if "." in doc else ""
            icon = FILE_ICONS.get(ext, "📄")
            st.markdown(f"### {icon} {doc}")
            st.write(summary)
            st.markdown("---")

    if not is_comparison_query(query) and st.session_state.show_sources and retrieved_chunks:
        with st.expander(f"📎 Retrieved Sources ({len(retrieved_chunks)})"):
            for item in retrieved_chunks:
                metadata = item.get("metadata", {})
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "?")
                score = item["score"]
                preview = item["text"][:500]
                st.markdown(
                    f"""
<div class="source-card">
  <div class="badge-row">
    <span class="badge badge-doc">📄 {source}</span>
    <span class="badge badge-page">p. {page}</span>
    <span class="badge badge-score">⚡ {score:.3f}</span>
  </div>
  <div class="source-preview">{preview}</div>
</div>""",
                    unsafe_allow_html=True,
                )


# =====================================================
# EXPORT
# =====================================================

if st.session_state.messages:
    st.markdown("")
    st.download_button(
        label="📥 Export Conversation",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="docmind_chat.json",
        mime="application/json",
    )
