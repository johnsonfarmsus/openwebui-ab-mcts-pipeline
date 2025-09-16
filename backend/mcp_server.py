"""
MCP Server for Open WebUI Integration

This server acts as a bridge between Open WebUI and our AB-MCTS and Multi-Model services,
allowing them to be used as tools within the Open WebUI interface.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
AB_MCTS_SERVICE_URL = "http://ab-mcts-service:8094"
MULTI_MODEL_SERVICE_URL = "http://multi-model-service:8090"

app = FastAPI(title="AB-MCTS & Multi-Model MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Protocol Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

# Available Tools
TOOLS = [
    MCPTool(
        name="ab_mcts_query",
        description="Run a query using AB-MCTS (Adaptive Branching Monte Carlo Tree Search) with multiple models for advanced reasoning and problem-solving",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or problem to solve using AB-MCTS"
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of search iterations (1-100)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100
                },
                "max_depth": {
                    "type": "integer", 
                    "description": "Maximum tree depth (1-20)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of models to use (optional, uses current selection if not provided)"
                }
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="multi_model_query",
        description="Run a query using Multi-Model collaboration where multiple AI models work together to provide comprehensive answers",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or problem to solve using multi-model collaboration"
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of models to use (optional, uses current selection if not provided)"
                }
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="ab_mcts_models",
        description="Get available models for AB-MCTS and update model selection",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "update"],
                    "description": "Action to perform: 'list' to get available models, 'update' to change selection"
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to select (required for 'update' action)"
                }
            },
            "required": ["action"]
        }
    ),
    MCPTool(
        name="multi_model_models",
        description="Get available models for Multi-Model collaboration and update model selection",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "update"],
                    "description": "Action to perform: 'list' to get available models, 'update' to change selection"
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to select (required for 'update' action)"
                }
            },
            "required": ["action"]
        }
    )
]

async def call_ab_mcts_service(query: str, iterations: int = 20, max_depth: int = 5, models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Call the AB-MCTS service."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "query": query,
                "iterations": iterations,
                "max_depth": max_depth
            }
            if models:
                payload["models"] = models
                
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"AB-MCTS service error: {e}")
        raise HTTPException(status_code=500, detail=f"AB-MCTS service error: {str(e)}")

async def call_multi_model_service(query: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Call the Multi-Model service."""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            payload = {"query": query}
            if models:
                payload["models"] = models
                
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Multi-Model service error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-Model service error: {str(e)}")

async def get_ab_mcts_models() -> Dict[str, Any]:
    """Get available AB-MCTS models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{AB_MCTS_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get AB-MCTS models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AB-MCTS models: {str(e)}")

async def update_ab_mcts_models(models: List[str]) -> Dict[str, Any]:
    """Update AB-MCTS model selection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/models/update",
                json={"models": models}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to update AB-MCTS models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update AB-MCTS models: {str(e)}")

async def get_multi_model_models() -> Dict[str, Any]:
    """Get available Multi-Model models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{MULTI_MODEL_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get Multi-Model models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Multi-Model models: {str(e)}")

async def update_multi_model_models(models: List[str]) -> Dict[str, Any]:
    """Update Multi-Model model selection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/models/update",
                json={"models": models}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to update Multi-Model models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update Multi-Model models: {str(e)}")

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP protocol requests."""
    try:
        if request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [tool.dict() for tool in TOOLS]
                }
            )
        
        elif request.method == "tools/call":
            if not request.params:
                raise HTTPException(status_code=400, detail="Missing parameters")
            
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name == "ab_mcts_query":
                result = await call_ab_mcts_service(
                    query=arguments["query"],
                    iterations=arguments.get("iterations", 20),
                    max_depth=arguments.get("max_depth", 5),
                    models=arguments.get("models")
                )
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"**AB-MCTS Result:**\n\n{result.get('result', 'No result')}\n\n**Search Statistics:**\n- Width Searches: {result.get('search_stats', {}).get('width_searches', 0)}\n- Depth Searches: {result.get('search_stats', {}).get('depth_searches', 0)}\n- Total Iterations: {result.get('search_stats', {}).get('total_iterations', 0)}\n- Response Time: {result.get('search_stats', {}).get('response_time', 0):.2f}s"
                            }
                        ]
                    }
                )
            
            elif tool_name == "multi_model_query":
                result = await call_multi_model_service(
                    query=arguments["query"],
                    models=arguments.get("models")
                )
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": f"**Multi-Model Result:**\n\n{result.get('result', 'No result')}\n\n**Quality Scores:**\n{json.dumps(result.get('search_stats', {}).get('quality_scores', {}), indent=2)}"
                            }
                        ]
                    }
                )
            
            elif tool_name == "ab_mcts_models":
                action = arguments["action"]
                if action == "list":
                    result = await get_ab_mcts_models()
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"**Available AB-MCTS Models:**\n\n{json.dumps(result.get('models', []), indent=2)}\n\n**Currently Selected:**\n{json.dumps(result.get('current_models', []), indent=2)}"
                                }
                            ]
                        }
                    )
                elif action == "update":
                    models = arguments["models"]
                    result = await update_ab_mcts_models(models)
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"**AB-MCTS Models Updated:**\n\n{json.dumps(result.get('current_models', []), indent=2)}"
                                }
                            ]
                        }
                    )
            
            elif tool_name == "multi_model_models":
                action = arguments["action"]
                if action == "list":
                    result = await get_multi_model_models()
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"**Available Multi-Model Models:**\n\n{json.dumps(result.get('models', []), indent=2)}\n\n**Currently Selected:**\n{json.dumps(result.get('current_models', []), indent=2)}"
                                }
                            ]
                        }
                    )
                elif action == "update":
                    models = arguments["models"]
                    result = await update_multi_model_models(models)
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"**Multi-Model Models Updated:**\n\n{json.dumps(result.get('current_models', []), indent=2)}"
                                }
                            ]
                        }
                    )
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
    
    except Exception as e:
        logger.error(f"MCP request error: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mcp-server"}

@app.get("/tools")
async def list_tools():
    """List available tools (for debugging)."""
    return {"tools": [tool.dict() for tool in TOOLS]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8096)
