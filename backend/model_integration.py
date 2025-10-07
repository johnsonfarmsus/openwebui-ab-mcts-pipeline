"""
Open WebUI Model Integration Service

This service exposes AB-MCTS and Multi-Model as selectable models
that appear in Open WebUI's model dropdown.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from typing import List, Dict, Any, Optional
import httpx
import uvicorn
from pydantic import BaseModel
import json
import time
import asyncio
import os
import sys
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
# Try absolute import via project root; fallback to module inside backend dir
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.services.experiment_logger import ExperimentLogger
except Exception:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from services.experiment_logger import ExperimentLogger

app = FastAPI(
    title="AB-MCTS & Multi-Model Models",
    version="1.0.0",
    description="Model integration for AB-MCTS and Multi-Model pipelines"
)

# Prometheus metrics
MODEL_INTEGRATION_REQUESTS = Counter(
    "model_integration_requests_total",
    "Total requests to model integration",
    ["model"]
)
MODEL_INTEGRATION_SUCCESS = Counter(
    "model_integration_success_total",
    "Successful model integration responses",
    ["model"]
)
MODEL_INTEGRATION_FAILURES = Counter(
    "model_integration_failures_total",
    "Failed model integration responses",
    ["model"]
)
MODEL_INTEGRATION_LATENCY = Histogram(
    "model_integration_latency_seconds",
    "Model integration latency in seconds",
    ["model"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)
ACTIVE_QUERIES_GAUGE = Gauge(
    "model_integration_active_queries",
    "Number of active queries"
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
OPENWEBUI_INTEGRATION_URL = "http://openwebui-integration:8097"

# Performance tracking
performance_stats = {
    "ab_mcts": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0,
        "last_request_time": None
    },
    "multi_model": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0,
        "last_request_time": None
    }
}

# Configuration storage (persisted to logs volume)
_CONFIG_PATH = os.getenv("MODEL_INTEGRATION_CONFIG_FILE", "/app/logs/model_integration_config.json")

def _default_config() -> Dict[str, Any]:
    return {
        "ab_mcts_iterations": 20,
        "ab_mcts_max_depth": 5,
        "auto_tools_enabled": True,
    }

def _load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(_CONFIG_PATH):
            import json as _json
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = _json.load(f)
                # basic validation
                data["ab_mcts_iterations"] = int(data.get("ab_mcts_iterations", 20))
                data["ab_mcts_max_depth"] = int(data.get("ab_mcts_max_depth", 5))
                data["auto_tools_enabled"] = bool(data.get("auto_tools_enabled", True))
                return data
    except Exception:
        pass
    return _default_config()

def _save_config(cfg: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        import json as _json
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
    except Exception:
        pass

configuration = _load_config()

# Query status tracking
active_queries = {}
query_counter = 0

# Run logger
experiment_logger = ExperimentLogger()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    iterations: Optional[int] = None
    max_depth: Optional[int] = None

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
    return {
        "status": "healthy",
        "service": "model-integration",
        "timestamp": time.time(),
        "timeout_settings": {
            "ab_mcts_timeout": "600 seconds",
            "multi_model_timeout": "300 seconds"
        },
        "optimized_parameters": {
            "ab_mcts_iterations": configuration["ab_mcts_iterations"],
            "ab_mcts_max_depth": configuration["ab_mcts_max_depth"],
            "auto_tools_enabled": configuration["auto_tools_enabled"],
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/performance")
async def get_performance_stats():
    """Get performance statistics."""
    return {
        "service": "model-integration",
        "timestamp": time.time(),
        "performance_stats": performance_stats,
        "uptime_seconds": time.time() - (performance_stats["ab_mcts"]["last_request_time"] or time.time())
    }

@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    return {
        "service": "model-integration",
        "configuration": configuration,
        "timestamp": time.time()
    }

@app.post("/config")
async def update_configuration(request: Dict[str, Any]):
    """Update configuration."""
    try:
        if "ab_mcts_iterations" in request:
            configuration["ab_mcts_iterations"] = max(1, min(100, int(request["ab_mcts_iterations"])))
        if "ab_mcts_max_depth" in request:
            configuration["ab_mcts_max_depth"] = max(1, min(20, int(request["ab_mcts_max_depth"])))
        if "auto_tools_enabled" in request:
            configuration["auto_tools_enabled"] = bool(request["auto_tools_enabled"])
        _save_config(configuration)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "configuration": configuration
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")

@app.get("/query-status")
async def get_query_status():
    """Get current query status for progress monitoring."""
    return {
        "service": "model-integration",
        "timestamp": time.time(),
        "active_queries": active_queries,
        "total_active": len(active_queries)
    }

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

    # Track metrics
    MODEL_INTEGRATION_REQUESTS.labels(model=request.model).inc()
    ACTIVE_QUERIES_GAUGE.inc()
    start_time = time.time()

    # Extract the user's message
    user_message = ""
    for message in request.messages:
        if message.role == "user":
            user_message = message.content
            break

    if not user_message:
        ACTIVE_QUERIES_GAUGE.dec()
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Use the user message as-is (enrichment moved to separate tool)
    enriched_message = user_message
    tool_context = ""
    tool_events: List[Dict[str, Any]] = []

    # Streaming path (OpenAI-compatible SSE stream)
    try:
        if getattr(request, "stream", False):
            if request.model == "ab-mcts":
                return StreamingResponse(
                    stream_ab_mcts(enriched_message, request, tool_context, tool_events),
                    media_type="text/event-stream",
                )
            elif request.model == "multi-model":
                return StreamingResponse(
                    stream_multi_model(enriched_message, request, tool_context, tool_events),
                    media_type="text/event-stream",
                )
            else:
                ACTIVE_QUERIES_GAUGE.dec()
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        # Non-streaming path
        if request.model == "ab-mcts":
            result = await call_ab_mcts(enriched_message, request)
        elif request.model == "multi-model":
            result = await call_multi_model(enriched_message, request)
        else:
            ACTIVE_QUERIES_GAUGE.dec()
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        # Record success metrics
        MODEL_INTEGRATION_SUCCESS.labels(model=request.model).inc()
        MODEL_INTEGRATION_LATENCY.labels(model=request.model).observe(time.time() - start_time)
        ACTIVE_QUERIES_GAUGE.dec()
        return result
    except Exception as e:
        MODEL_INTEGRATION_FAILURES.labels(model=request.model).inc()
        ACTIVE_QUERIES_GAUGE.dec()
        raise

def build_openai_stream_chunk(content: str, model: str, include_role: bool = False, finish_reason: Optional[str] = None) -> str:
    """Build an OpenAI-compatible streaming chunk line (Server-Sent Event)."""
    delta: Dict[str, Any] = {"content": content}
    if include_role:
        delta["role"] = "assistant"
    chunk = {
        "id": f"chatcmpl-chunk-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"

async def stream_ab_mcts(user_message: str, request: ChatRequest, tool_context: str = "", tool_events: Optional[List[Dict[str, Any]]] = None):
    """Stream AB-MCTS progress and final result to keep UI connection alive."""
    model_name = "ab-mcts"
    # Resolve parameters: use request if provided, else configuration
    resolved_iterations = request.iterations if request.iterations is not None else configuration["ab_mcts_iterations"]
    resolved_max_depth = request.max_depth if request.max_depth is not None else configuration["ab_mcts_max_depth"]
    
    # Track active query for dashboard auto-monitor
    global query_counter
    query_counter += 1
    query_id = f"ab_mcts_{query_counter}_{int(time.time())}"
    start_time = time.time()
    active_queries[query_id] = {
        "model": "ab-mcts",
        "status": "initializing",
        "start_time": start_time,
        "query_preview": user_message[:50] + "..." if len(user_message) > 50 else user_message,
        "iterations": resolved_iterations,
        "max_depth": resolved_max_depth
    }
    # Start run log
    run_id = experiment_logger.start_run(
        pipeline="ab-mcts",
        user_query=user_message,
        parameters={"iterations": resolved_iterations, "max_depth": resolved_max_depth},
        metadata={"query_id": query_id},
    )
    experiment_logger.log_event(run_id, {"type": "status", "status": "initializing"})

    # Initial role chunk
    yield build_openai_stream_chunk("", model_name, include_role=True)
    # Initial notice
    intro = (
        f"Starting AB-MCTS... iterations={resolved_iterations}, max_depth={resolved_max_depth}\n"
    )
    yield build_openai_stream_chunk(intro, model_name)
    if tool_context:
        yield build_openai_stream_chunk("Tool context detected. Summaries included below.\n", model_name)
        # show compact head only
        preview = tool_context[:800]
        yield build_openai_stream_chunk(preview + "\n", model_name)

    # Kick off background request
    async with httpx.AsyncClient(timeout=600.0) as client:
        payload = {
            "query": user_message,
            "iterations": resolved_iterations,
            "max_depth": resolved_max_depth,
        }
        post_task = asyncio.create_task(
            client.post(f"{AB_MCTS_SERVICE_URL}/query", json=payload)
        )
        experiment_logger.log_event(run_id, {"type": "request_sent", "payload": {k: (v if k != 'query' else str(v)[:400]) for k, v in payload.items()}})
        for ev in (tool_events or []):
            # Telemetry: mark source and confidence for enrichment events
            if ev.get("type") == "tool":
                ev.setdefault("source", "auto_enrich")
                ev.setdefault("confidence", "high" if ev.get("ok") else "low")
            experiment_logger.log_event(run_id, ev)

        # Periodic keepalive/progress pings
        elapsed = 0
        interval = 25
        try:
            while not post_task.done():
                # Update dashboard status
                if query_id in active_queries:
                    if elapsed < 25:
                        active_queries[query_id]["status"] = "building_tree"
                    elif elapsed < 50:
                        active_queries[query_id]["status"] = "querying_models"
                    elif elapsed < 75:
                        active_queries[query_id]["status"] = "evaluating_solutions"
                    else:
                        active_queries[query_id]["status"] = "selecting_best_result"
                experiment_logger.log_event(run_id, {"type": "status", "status": active_queries[query_id]["status"], "elapsed": elapsed})

                msg = "\n⏳ AB-MCTS is still working...\n"
                yield build_openai_stream_chunk(msg, model_name)
                await asyncio.sleep(interval)
                elapsed += interval

            # Completed
            response = await post_task
            response.raise_for_status()
            data = response.json()
            experiment_logger.log_event(run_id, {"type": "response_received", "data_head": str(data)[:400]})
            result = data.get("result", "No response from AB-MCTS")

            # Stream final result in chunks
            # Split into manageable pieces to avoid huge single chunk
            chunk_size = 800
            for i in range(0, len(result), chunk_size):
                yield build_openai_stream_chunk(result[i:i+chunk_size], model_name)

            # Send finish signal chunk and DONE sentinel
            yield build_openai_stream_chunk("", model_name, finish_reason="stop")
            yield "data: [DONE]\n\n"
            # Clean up active query
            if query_id in active_queries:
                del active_queries[query_id]
            experiment_logger.finish_run(run_id, {"result": result, "search_stats": data.get("search_stats", {})})
        except httpx.HTTPError as e:
            err_msg = f"\n❌ AB-MCTS HTTP error: {str(e)}\n"
            yield build_openai_stream_chunk(err_msg, model_name)
            yield build_openai_stream_chunk("", model_name, finish_reason="error")
            yield "data: [DONE]\n\n"
            if query_id in active_queries:
                del active_queries[query_id]
            experiment_logger.fail_run(run_id, f"http_error: {str(e)}")
        except Exception as e:
            err_msg = f"\n❌ AB-MCTS error: {str(e)}\n"
            yield build_openai_stream_chunk(err_msg, model_name)
            yield build_openai_stream_chunk("", model_name, finish_reason="error")
            yield "data: [DONE]\n\n"
            if query_id in active_queries:
                del active_queries[query_id]
            experiment_logger.fail_run(run_id, f"error: {str(e)}")

async def stream_multi_model(user_message: str, request: ChatRequest, tool_context: str = "", tool_events: Optional[List[Dict[str, Any]]] = None):
    """Stream Multi-Model progress and final result."""
    model_name = "multi-model"
    yield build_openai_stream_chunk("", model_name, include_role=True)
    yield build_openai_stream_chunk("Starting Multi-Model collaboration...\n", model_name)
    if tool_context:
        yield build_openai_stream_chunk("Tool context detected. Summaries included below.\n", model_name)
        preview = tool_context[:800]
        yield build_openai_stream_chunk(preview + "\n", model_name)

    # Start run log
    run_id = experiment_logger.start_run(
        pipeline="multi-model",
        user_query=user_message,
        parameters={},
        metadata={},
    )
    experiment_logger.log_event(run_id, {"type": "status", "status": "initializing"})

    async with httpx.AsyncClient(timeout=300.0) as client:
        payload = {"query": user_message}
        post_task = asyncio.create_task(
            client.post(f"{MULTI_MODEL_SERVICE_URL}/query", json=payload)
        )
        experiment_logger.log_event(run_id, {"type": "request_sent", "payload": {k: (v if k != 'query' else str(v)[:400]) for k, v in payload.items()}})
        for ev in (tool_events or []):
            experiment_logger.log_event(run_id, ev)
        elapsed = 0
        interval = 25
        try:
            while not post_task.done():
                msg = "\n⏳ Collaborating across models...\n"
                yield build_openai_stream_chunk(msg, model_name)
                await asyncio.sleep(interval)
                elapsed += interval

            response = await post_task
            response.raise_for_status()
            data = response.json()
            experiment_logger.log_event(run_id, {"type": "response_received", "data_head": str(data)[:400]})
            result = data.get("result", "No response from Multi-Model")
            chunk_size = 800
            for i in range(0, len(result), chunk_size):
                yield build_openai_stream_chunk(result[i:i+chunk_size], model_name)
            yield build_openai_stream_chunk("", model_name, finish_reason="stop")
            yield "data: [DONE]\n\n"
            experiment_logger.finish_run(run_id, {"result": result, "search_stats": data.get("search_stats", {})})
        except httpx.HTTPError as e:
            err_msg = f"\n❌ Multi-Model HTTP error: {str(e)}\n"
            yield build_openai_stream_chunk(err_msg, model_name)
            yield build_openai_stream_chunk("", model_name, finish_reason="error")
            yield "data: [DONE]\n\n"
            experiment_logger.fail_run(run_id, f"http_error: {str(e)}")
        except Exception as e:
            err_msg = f"\n❌ Multi-Model error: {str(e)}\n"
            yield build_openai_stream_chunk(err_msg, model_name)
            yield build_openai_stream_chunk("", model_name, finish_reason="error")
            yield "data: [DONE]\n\n"
            experiment_logger.fail_run(run_id, f"error: {str(e)}")

async def call_ab_mcts(user_message: str, request: ChatRequest):
    """Call AB-MCTS service with optimized parameters."""
    global query_counter
    query_counter += 1
    query_id = f"ab_mcts_{query_counter}_{int(time.time())}"
    
    start_time = time.time()
    performance_stats["ab_mcts"]["total_requests"] += 1
    performance_stats["ab_mcts"]["last_request_time"] = start_time
    
    # Track active query
    active_queries[query_id] = {
        "model": "ab-mcts",
        "status": "initializing",
        "start_time": start_time,
        "query_preview": user_message[:50] + "..." if len(user_message) > 50 else user_message,
        "iterations": request.iterations if hasattr(request, 'iterations') and request.iterations else configuration["ab_mcts_iterations"],
        "max_depth": request.max_depth if hasattr(request, 'max_depth') and request.max_depth else configuration["ab_mcts_max_depth"]
    }
    
    try:
        # Update status to building tree
        active_queries[query_id]["status"] = "building_tree"
        
        # Increase timeout to 600 seconds and reduce iterations for faster response
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Update status to querying models
            active_queries[query_id]["status"] = "querying_models"
            
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/query",
                json={
                    "query": user_message,
                    "iterations": request.iterations if hasattr(request, 'iterations') and request.iterations else configuration["ab_mcts_iterations"],
                    "max_depth": request.max_depth if hasattr(request, 'max_depth') and request.max_depth else configuration["ab_mcts_max_depth"]
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Update status to evaluating solutions
            active_queries[query_id]["status"] = "evaluating_solutions"
            
            # Update performance stats
            response_time = time.time() - start_time
            performance_stats["ab_mcts"]["successful_requests"] += 1
            performance_stats["ab_mcts"]["average_response_time"] = (
                (performance_stats["ab_mcts"]["average_response_time"] * (performance_stats["ab_mcts"]["successful_requests"] - 1) + response_time) 
                / performance_stats["ab_mcts"]["successful_requests"]
            )
            
            # Update status to selecting best result
            active_queries[query_id]["status"] = "selecting_best_result"
            
            # Basic response quality validation
            result = data.get("result", "No response from AB-MCTS")
            if result and len(result) > 10:  # Basic validation
                # Check for obvious error patterns
                if "Error:" in result or "No response" in result:
                    result = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or try again later."
            
            # Clean up completed query
            if query_id in active_queries:
                del active_queries[query_id]
            
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
                        "content": result
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(data.get("result", "").split()),
                    "total_tokens": len(user_message.split()) + len(data.get("result", "").split())
                }
            }
    except httpx.TimeoutException:
        performance_stats["ab_mcts"]["failed_requests"] += 1
        # Clean up failed query
        if query_id in active_queries:
            del active_queries[query_id]
        print("AB-MCTS service timeout - query took too long")
        raise HTTPException(
            status_code=408, 
            detail="AB-MCTS query timed out. The tree search is taking longer than expected. Please try a simpler query or try again later."
        )
    except httpx.ConnectError:
        performance_stats["ab_mcts"]["failed_requests"] += 1
        # Clean up failed query
        if query_id in active_queries:
            del active_queries[query_id]
        print("AB-MCTS service connection error")
        raise HTTPException(
            status_code=503, 
            detail="AB-MCTS service is unavailable. Please check if the service is running."
        )
    except Exception as e:
        performance_stats["ab_mcts"]["failed_requests"] += 1
        # Clean up failed query
        if query_id in active_queries:
            del active_queries[query_id]
        print(f"AB-MCTS service error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"AB-MCTS service error: {str(e)}. Please try again or contact support if the issue persists."
        )

async def call_multi_model(user_message: str, request: ChatRequest):
    """Call Multi-Model service with increased timeout."""
    start_time = time.time()
    performance_stats["multi_model"]["total_requests"] += 1
    performance_stats["multi_model"]["last_request_time"] = start_time
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/query",
                json={
                    "query": user_message
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Update performance stats
            response_time = time.time() - start_time
            performance_stats["multi_model"]["successful_requests"] += 1
            performance_stats["multi_model"]["average_response_time"] = (
                (performance_stats["multi_model"]["average_response_time"] * (performance_stats["multi_model"]["successful_requests"] - 1) + response_time) 
                / performance_stats["multi_model"]["successful_requests"]
            )
            
            # Basic response quality validation
            result = data.get("result", "No response from Multi-Model")
            if result and len(result) > 10:  # Basic validation
                # Check for obvious error patterns
                if "Error:" in result or "No response" in result:
                    result = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or try again later."
            
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
                        "content": result
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(data.get("result", "").split()),
                    "total_tokens": len(user_message.split()) + len(data.get("result", "").split())
                }
            }
    except httpx.TimeoutException:
        performance_stats["multi_model"]["failed_requests"] += 1
        print("Multi-Model service timeout - query took too long")
        raise HTTPException(
            status_code=408, 
            detail="Multi-Model query timed out. The collaboration process is taking longer than expected. Please try again later."
        )
    except httpx.ConnectError:
        performance_stats["multi_model"]["failed_requests"] += 1
        print("Multi-Model service connection error")
        raise HTTPException(
            status_code=503, 
            detail="Multi-Model service is unavailable. Please check if the service is running."
        )
    except Exception as e:
        performance_stats["multi_model"]["failed_requests"] += 1
        print(f"Multi-Model service error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Multi-Model service error: {str(e)}. Please try again or contact support if the issue persists."
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8098)
