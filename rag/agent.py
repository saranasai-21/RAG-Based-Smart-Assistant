from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from rag.llm import load_llm
from rag.prompts import build_qa_prompt
from rag.logging_config import get_logger
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

class AgentState(TypedDict):
    query: str
    context: str
    sources_text: str
    history_text: str
    generation: str
    confidence_score: float
    hallucination: bool
    iterations: int

def retrieve(state: AgentState):
    """Retrieval is handled externally before passing to the agent for streaming simplicity,
    but this node logs the action."""
    logger.info("---RETRIEVE---")
    return {"iterations": state.get("iterations", 0) + 1}

def generate(state: AgentState):
    """Generate the answer using the MultiLLMRouter."""
    logger.info("---GENERATE---")
    prompt = build_qa_prompt(
        context=state["context"],
        sources_text=state["sources_text"],
        query=state["query"],
        history_text=state["history_text"]
    )
    llm = load_llm()
    try:
        # For agentic flow we do a single invoke to get the draft
        response = llm.invoke([HumanMessage(content=prompt)])
        generation = response.content
    except Exception as e:
        generation = f"Error during generation: {e}"
        
    # Calculate confidence based on context availability
    context_len = len(state.get("context", ""))
    if context_len > 2000:
        confidence = 0.92
    elif context_len > 500:
        confidence = 0.78
    elif context_len > 100:
        confidence = 0.55
    else:
        confidence = 0.3
        
    return {"generation": generation, "confidence_score": confidence}

def grade_hallucination(state: AgentState):
    """Check if the generation is grounded in the document."""
    logger.info("---CHECK HALLUCINATION---")
    # Simplistic mock hallucination check to avoid extra LLM call latency in this demo
    generation = state.get("generation", "")
    if "Error" in generation:
        return {"hallucination": True}
    return {"hallucination": False}

def route_hallucination(state: AgentState):
    """Route to refine or end."""
    if state["hallucination"] and state["iterations"] < 2:
        return "generate"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("generate", generate)
workflow.add_node("grade_hallucination", grade_hallucination)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "grade_hallucination")
workflow.add_conditional_edges(
    "grade_hallucination",
    route_hallucination,
    {
        "generate": "generate",
        "end": END
    }
)

agent_app = workflow.compile()
