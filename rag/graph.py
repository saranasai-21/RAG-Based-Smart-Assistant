import json
from typing import Dict, TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm import load_llm
from rag.retrieval import hybrid_search, format_sources
from rag.logging_config import get_logger

logger = get_logger(__name__)

class GraphState(TypedDict):
    """The state of the RAG workflow."""
    question: str
    chat_history: List[Dict[str, str]]
    intent: str
    expanded_queries: List[str]
    retrieved_chunks: List[Dict]
    generation: str
    sources: List[Dict]
    global_state: Dict

def intent_node(state: GraphState):
    """Classify the user intent."""
    logger.info("---DETECT INTENT---")
    question = state["question"]
    llm = load_llm()
    
    prompt = f"""You are an intent classifier for a document intelligence assistant.
    Classify the user's question into one of the following categories:
    - summarize: The user is asking for a general summary or overview of the documents.
    - qa: The user is asking a specific question.
    - comparison: The user is asking to compare things.
    
    Return ONLY the category name.
    
    Question: {question}
    Category:"""
    
    intent = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    if intent not in ["summarize", "qa", "comparison"]:
        intent = "qa"
        
    return {"intent": intent}

def expansion_node(state: GraphState):
    """Rewrite and expand the query."""
    logger.info("---EXPAND QUERY---")
    question = state["question"]
    intent = state["intent"]
    llm = load_llm()
    
    if intent == "summarize":
        return {"expanded_queries": [question]}
        
    prompt = f"""You are an expert search query generator.
    Given the user's question, generate 3 semantic variations of the question to improve search retrieval.
    Return ONLY the queries separated by newlines.
    
    Question: {question}"""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    queries = [q.strip() for q in response.split("\n") if q.strip()]
    queries.append(question)
    
    return {"expanded_queries": list(set(queries))}

def retrieve_node(state: GraphState):
    """Retrieve documents using Hybrid Search."""
    logger.info("---RETRIEVE---")
    expanded_queries = state["expanded_queries"]
    global_state = state["global_state"]
    
    all_results = []
    for q in expanded_queries:
        res = hybrid_search(
            query=q,
            vector_db=global_state.get("vector_db"),
            bm25=global_state.get("bm25"),
            chunks=global_state.get("chunks", []),
            metadata_list=global_state.get("metadata", []),
            k=5
        )
        all_results.extend(res)
        
    # Deduplicate by text
    seen = set()
    deduped = []
    for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
        if r["text"] not in seen:
            seen.add(r["text"])
            deduped.append(r)
            
    # Top 5 overall
    top_chunks = deduped[:5]
    return {"retrieved_chunks": top_chunks}

def summarize_node(state: GraphState):
    """Generate a holistic summary."""
    logger.info("---SUMMARIZE---")
    global_state = state["global_state"]
    chunks = global_state.get("chunks", [])
    
    # Combine chunks up to token limit
    full_text = "\\n\\n".join(chunks)[:100000] # Gemini 1.5 flash has huge context
    
    llm = load_llm()
    prompt = f"""You are an advanced document intelligence assistant.
    The user has asked for a summary of the uploaded documents.
    Create a comprehensive summary including:
    - Executive Summary
    - Key Topics
    - Important Concepts
    - Main Findings
    - Conclusion
    
    Documents:
    {full_text}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": response.content, "sources": []}

def generate_node(state: GraphState):
    """Generate answer with sources and guardrails."""
    logger.info("---GENERATE---")
    question = state["question"]
    chunks = state["retrieved_chunks"]
    chat_history = state["chat_history"]
    llm = load_llm()
    
    if not chunks:
        return {"generation": "I could not find enough evidence in the uploaded documents.", "sources": []}
        
    context = ""
    sources = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown Document")
        page = meta.get("page", "?")
        context += f"Source [{idx+1}] ({source}, Page {page}):\\n{chunk['text']}\\n\\n"
        sources.append({"file": source, "page": page, "relevance": round(chunk.get("score", 0), 2)})
        
    history_str = ""
    for turn in chat_history[-4:]:
        history_str += f"User: {turn.get('user', '')}\\nAssistant: {turn.get('assistant', '')}\\n"
        
    system_prompt = """You are an advanced document intelligence assistant.
    Always answer using the provided context.
    
    Rules:
    1. First answer the user's question directly.
    2. Use retrieved context as evidence.
    3. If information is unavailable, say: "The uploaded documents do not contain enough information."
    4. Never say: "You did not provide a question" if a user message exists.
    5. Use markdown formatting.
    6. Cite source pages like [1].
    
    Context:
    {context}
    
    Chat History:
    {history}
    """
    
    sys_msg = SystemMessage(content=system_prompt.format(context=context, history=history_str))
    usr_msg = HumanMessage(content=f"User Question: {question}\\n\\nAnswer:")
    
    response = llm.invoke([sys_msg, usr_msg])
    
    # Simple hallucination check via LLM output
    gen = response.content
    if "do not contain enough information" in gen.lower():
        sources = []
        
    return {"generation": gen, "sources": sources}

def route_intent(state: GraphState):
    """Route to summarize or retrieve based on intent."""
    if state["intent"] == "summarize":
        return "summarize"
    return "retrieve"

def create_rag_graph():
    """Create the compiled LangGraph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("intent", intent_node)
    workflow.add_node("expansion", expansion_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "expansion")
    workflow.add_conditional_edges(
        "expansion",
        route_intent,
        {
            "summarize": "summarize",
            "retrieve": "retrieve"
        }
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("summarize", END)
    workflow.add_edge("generate", END)
    
    return workflow.compile()
