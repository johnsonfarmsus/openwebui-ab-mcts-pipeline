"""
Open WebUI Model Integration Service

This service exposes AB-MCTS and Multi-Model as selectable models
that appear in Open WebUI's model dropdown.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import httpx
import uvicorn
from pydantic import BaseModel
import json

app = FastAPI(
    title="AB-MCTS & Multi-Model Models",
    version="1.0.0",
    description="Model integration for AB-MCTS and Multi-Model pipelines"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
AB_MCTS_SERVICE_URL = "http://ab-mcts-service:8094"
MULTI_MODEL_SERVICE_URL = "http://multi-model-service:8090"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]]
    root: str
    parent: Optional[str] = None

# Model definitions
MODELS = {
    "ab-mcts": {
        "id": "ab-mcts",
        "name": "AB-MCTS (Advanced Tree Search)",
        "description": "Adaptive Branching Monte Carlo Tree Search for complex problem solving",
        "capabilities": ["reasoning", "problem_solving", "tree_search"]
    },
    "multi-model": {
        "id": "multi-model", 
        "name": "Multi-Model (Collaborative AI)",
        "description": "Multiple AI models working together for comprehensive answers",
        "capabilities": ["collaboration", "comprehensive_analysis", "multi_perspective"]
    }
}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AB-MCTS & Multi-Model Model Integration Service"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "model-integration"}

@app.get("/v1/models")
async def list_models():
    """List available models - this is what Open WebUI calls to discover models."""
    models = []
    for model_id, model_info in MODELS.items():
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1700000000,  # Fixed timestamp
            "owned_by": "ab-mcts-multi-model",
            "permission": [],
            "root": model_id,
            "parent": None
        })
    return {"object": "list", "data": models}

@app.get("/models")
async def list_models_alt():
    """Alternative models endpoint for Open WebUI compatibility."""
    return await list_models()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions - this is what Open WebUI calls when a model is selected."""
    return await handle_chat_completions(request)

@app.post("/chat/completions")
async def chat_completions_alt(request: ChatRequest):
    """Alternative chat completions endpoint for Open WebUI compatibility."""
    return await handle_chat_completions(request)

async def handle_chat_completions(request: ChatRequest):
    """Handle chat completions logic."""
    
    # Extract the user's message
    user_message = ""
    for message in request.messages:
        if message.role == "user":
            user_message = message.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Route to the appropriate service based on model
    if request.model == "ab-mcts":
        return await call_ab_mcts(user_message, request)
    elif request.model == "multi-model":
        return await call_multi_model(user_message, request)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

async def call_ab_mcts(user_message: str, request: ChatRequest):
    """Call AB-MCTS service."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/query",
                json={
                    "query": user_message,
                    "iterations": 20,
                    "max_depth": 5
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Format response for Open WebUI
            return {
                "id": f"chatcmpl-{hash(user_message) % 1000000}",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "ab-mcts",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("result", "No response from AB-MCTS")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(data.get("result", "").split()),
                    "total_tokens": len(user_message.split()) + len(data.get("result", "").split())
                }
            }
    except Exception as e:
        print(f"AB-MCTS service error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AB-MCTS service error: {str(e)}")

async def call_multi_model(user_message: str, request: ChatRequest):
    """Call Multi-Model service."""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/query",
                json={
                    "query": user_message
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Format response for Open WebUI
            return {
                "id": f"chatcmpl-{hash(user_message) % 1000000}",
                "object": "chat.completion", 
                "created": 1700000000,
                "model": "multi-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("result", "No response from Multi-Model")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(data.get("result", "").split()),
                    "total_tokens": len(user_message.split()) + len(data.get("result", "").split())
                }
            }
    except Exception as e:
        print(f"Multi-Model service error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-Model service error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8098)
