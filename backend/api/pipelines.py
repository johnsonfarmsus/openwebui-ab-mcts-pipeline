"""
Pipeline management endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import requests
import asyncio
from datetime import datetime

router = APIRouter()

# Service endpoints
AB_MCTS_SERVICE_URL = "http://ab-mcts-service:8094"
MULTI_MODEL_SERVICE_URL = "http://multi-model-service:8090"

@router.get("/status")
async def get_pipeline_status():
    """Get status of all pipelines."""
    status = {}
    
    # Check AB-MCTS service
    try:
        response = requests.get(f"{AB_MCTS_SERVICE_URL}/health", timeout=5)
        status["ab_mcts"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": AB_MCTS_SERVICE_URL
        }
    except Exception as e:
        status["ab_mcts"] = {
            "status": "unhealthy",
            "error": str(e),
            "url": AB_MCTS_SERVICE_URL
        }
    
    # Check Multi-Model service
    try:
        response = requests.get(f"{MULTI_MODEL_SERVICE_URL}/health", timeout=5)
        status["multi_model"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": MULTI_MODEL_SERVICE_URL
        }
    except Exception as e:
        status["multi_model"] = {
            "status": "unhealthy",
            "error": str(e),
            "url": MULTI_MODEL_SERVICE_URL
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "pipelines": status
    }

@router.post("/ab-mcts/query")
async def ab_mcts_query(query_data: Dict[str, Any]):
    """Process query using AB-MCTS pipeline."""
    try:
        response = requests.post(
            f"{AB_MCTS_SERVICE_URL}/query",
            json=query_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"AB-MCTS service error: {str(e)}")

@router.post("/multi-model/query")
async def multi_model_query(query_data: Dict[str, Any]):
    """Process query using Multi-Model pipeline."""
    try:
        response = requests.post(
            f"{MULTI_MODEL_SERVICE_URL}/query",
            json=query_data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Multi-Model service error: {str(e)}")

@router.get("/ab-mcts/stats")
async def get_ab_mcts_stats():
    """Get AB-MCTS pipeline statistics."""
    try:
        response = requests.get(f"{AB_MCTS_SERVICE_URL}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AB-MCTS stats: {str(e)}")

@router.get("/multi-model/stats")
async def get_multi_model_stats():
    """Get Multi-Model pipeline statistics."""
    try:
        response = requests.get(f"{MULTI_MODEL_SERVICE_URL}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Multi-Model stats: {str(e)}")

@router.post("/ab-mcts/restart")
async def restart_ab_mcts(background_tasks: BackgroundTasks):
    """Restart AB-MCTS service."""
    # This would typically involve Docker commands or service management
    background_tasks.add_task(restart_service, "ab-mcts")
    return {"message": "AB-MCTS service restart initiated"}

@router.post("/multi-model/restart")
async def restart_multi_model(background_tasks: BackgroundTasks):
    """Restart Multi-Model service."""
    background_tasks.add_task(restart_service, "multi-model")
    return {"message": "Multi-Model service restart initiated"}

async def restart_service(service_name: str):
    """Background task to restart a service."""
    # Implementation would depend on your deployment method
    # For Docker Compose, this might involve:
    # subprocess.run(["docker-compose", "restart", service_name])
    pass
