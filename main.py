# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from agents.graph import rag_agent
from graudrails import initialize_rails, guard
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()# initliaze gaurdrails the moment application started 

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    # Determine if this thread already has conversation history
    # so guardrails can combine previous context on follow-up cues for Gate 4.
    has_history = False
    last_user_question = ""
    history_text = ""
    try:
        checkpoint = rag_agent.checkpointer.get(config)
        msgs = checkpoint.get("channel_values", {}).get("messages", []) if checkpoint else []
        if msgs:
            has_history = True
            for m in msgs:
                history_text += f"{m.get('role', '')}: {m.get('content', '')}\n"
                if m.get("role") == "user":
                    last_user_question = m.get("content", "")
    except Exception:
        pass  # First message or checkpointer unavailable — treat as no history
    
    try:
        # Gate 1: Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(
            q,
            has_history=has_history,
            last_user_question=last_user_question,
            history_text=history_text
        )
        if rail_fired:
            # Distinguish dialog responses (Gate 1) from actual blocks (Gate 2/3/4)
            from graudrails.colang_rules import GREETING_RESPONSE, FAREWELL_RESPONSE, CAPABILITIES_RESPONSE
            dialog_responses = {GREETING_RESPONSE, FAREWELL_RESPONSE, CAPABILITIES_RESPONSE}
            if rail_response in dialog_responses:
                logfire.info(f"🛡️ Gate 1 dialog response | thread={thread_id}")
            else:
                logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
