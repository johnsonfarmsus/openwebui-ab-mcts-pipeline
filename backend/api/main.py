"""
Main FastAPI application for backend management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.pipelines import router as pipelines_router
from backend.api.monitoring import router as monitoring_router
from backend.api.models import router as models_router
from backend.api.config import router as config_router

# New: runs endpoints
from fastapi import APIRouter
from backend.services.experiment_logger import ExperimentLogger

# Create FastAPI app
app = FastAPI(
    title="AB-MCTS & Multi-Model Backend API",
    description="Backend management API for AB-MCTS and Multi-Model pipelines",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pipelines_router, prefix="/api/pipelines", tags=["pipelines"])
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(config_router, prefix="/api/config", tags=["configuration"])

# Runs API
runs_router = APIRouter()
logger = ExperimentLogger()

@runs_router.get("/")
async def list_runs(limit: int = 100):
    return {"runs": logger.list_runs(limit=limit)}

@runs_router.get("/{run_id}")
async def get_run(run_id: str):
    run = logger.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@runs_router.get("/{run_id}/events")
async def get_run_events(run_id: str, head: int | None = 200):
    return {"events": logger.read_events(run_id, head=head)}

app.include_router(runs_router, prefix="/api/runs", tags=["runs"])

# Mount static files for dashboard
dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "static")
if os.path.exists(dashboard_path):
    app.mount("/static", StaticFiles(directory=dashboard_path), name="static")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AB-MCTS & Multi-Model Backend API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend-api"}

@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    return {
        "status": "operational",
        "services": {
            "ab_mcts": "running",
            "multi_model": "running",
            "backend_api": "running"
        },
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8095)
