import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time
from src.pipeline.rag_pipeline import RAGPipeline
from configs.config import config

# Initialize app
app = FastAPI(
    title="RAG Analytics Chatbot",
    description="LLM-powered chatbot for natural language business KPI queries using RAG",
    version="1.0.0",
)

# Load pipeline once at startup
pipeline = None


@app.on_event("startup")
async def load_pipeline():
    global pipeline
    print("Loading RAG pipeline...")
    pipeline = RAGPipeline()
    print("Pipeline ready.")


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    num_sources: int
    retrieval_status: str
    latency_ms: float
    timestamp: str


class ConversationResponse(BaseModel):
    turns: int
    history: List[dict]


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "service": "RAG Analytics Chatbot",
        "version": "1.0.0",
        "status": "running",
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a natural language question about business KPIs."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = pipeline.query(request.query, top_k=request.top_k)
    return result


@app.get("/conversation", response_model=ConversationResponse)
def get_conversation():
    """Get the full conversation history."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    history = pipeline.get_conversation_history()
    return {
        "turns": len(history) // 2,
        "history": history,
    }


@app.delete("/conversation")
def clear_conversation():
    """Clear the conversation history."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    pipeline.clear_history()
    return {"status": "conversation cleared"}


@app.post("/conversation/save")
def save_conversation():
    """Save the conversation to disk."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    path = pipeline.save_conversation()
    return {"status": "saved", "path": path}


@app.get("/documents")
def get_documents():
    """List all documents in the vector store."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    docs = pipeline.vector_store.get_all_documents()
    return {
        "total_documents": len(docs),
        "documents": [{"id": d["id"], "title": d["title"], "category": d["category"]} for d in docs],
    }