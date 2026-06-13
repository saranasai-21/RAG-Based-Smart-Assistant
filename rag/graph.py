from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import os

from rag.llm import load_llm, detect_relevant_documents, generate_multi_queries, rewrite_followup_query
from rag.query_classification import is_followup_query
from rag.retrieval import hybrid_search, filter_results_by_documents, rerank_results, format_sources
from rag.prompts import build_qa_prompt
from langchain_core.messages import HumanMessage

class GraphState(TypedDict):
    query: str
    history: List[Dict[str, str]]
    expanded_queries: List[str]
    relevant_docs: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    response: str
    sources: List[str]

    # Global references for retrieval
    vector_db: Any
    bm25: Any
    chunks: List[str]
    metadata_list: List[Dict[str, Any]]
    document_summaries: Dict[str, str]

def analyze_query_node(state: GraphState) -> GraphState:
    query = state["query"]
    history = state["history"]
    llm = load_llm(api_key=os.getenv("GEMINI_API_KEY", ""))
    
    if is_followup_query(query) and history:
        query = rewrite_followup_query(llm, query, history)
        
    relevant_docs = detect_relevant_documents(query, state["document_summaries"], llm)
    return {"query": query, "relevant_docs": relevant_docs}

def expand_query_node(state: GraphState) -> GraphState:
    llm = load_llm(api_key=os.getenv("GEMINI_API_KEY", ""))
    queries = generate_multi_queries(llm, state["query"])
    queries.append(state["query"])
    return {"expanded_queries": list(set(queries))}

def retrieve_node(state: GraphState) -> GraphState:
    all_results = []
    for q in state["expanded_queries"]:
        res = hybrid_search(
            q, 
            state["vector_db"], 
            state["bm25"], 
            state["chunks"], 
            state["metadata_list"]
        )
        all_results.extend(res)
        
    filtered = filter_results_by_documents(all_results, state["relevant_docs"])
    
    seen = set()
    unique_res = []
    for r in filtered:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_res.append(r)
            
    return {"retrieved_chunks": unique_res}

def rerank_node(state: GraphState) -> GraphState:
    ranked = rerank_results(state["query"], state["retrieved_chunks"])
    return {"reranked_chunks": ranked[:10]} # Top 10

def generate_node(state: GraphState) -> GraphState:
    context = "\n\n---\n\n".join([r["text"] for r in state["reranked_chunks"]])
    prompt = build_qa_prompt(context, "", state["query"], str(state["history"]))
    
    llm = load_llm(api_key=os.getenv("GEMINI_API_KEY", ""))
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Format sources
    sources = []
    for r in state["reranked_chunks"]:
        md = r.get("metadata", {})
        s = md.get("source", "Unknown Document")
        p = md.get("page", "?")
        sources.append(f"{s} (Page {p})")
        
    return {"response": response.content, "sources": list(set(sources))}

def build_advanced_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("analyze", analyze_query_node)
    workflow.add_node("expand", expand_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "expand")
    workflow.add_edge("expand", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
