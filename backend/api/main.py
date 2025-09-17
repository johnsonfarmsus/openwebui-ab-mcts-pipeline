"""
Main FastAPI application for backend management.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Request
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

# --- Security & Rate Limiting ---
# API key check: set BACKEND_API_KEY env var to enforce; if empty, allow all
async def verify_api_key(x_api_key: str | None = Header(None)):
    expected_key = os.getenv("BACKEND_API_KEY", "").strip()
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# Simple per-IP, per-path rate limit (best-effort, in-memory)
_RATE_LIMIT_BUCKET: dict[tuple[str, str], list[float]] = {}
_RATE_LIMIT_MAX = int(os.getenv("BACKEND_RATE_LIMIT_PER_MIN", "120"))

async def rate_limit(request: Request):
    try:
        client_ip = request.client.host if request.client else "unknown"
    except Exception:
        client_ip = "unknown"
    path = request.url.path
    key = (client_ip, path)
    now = time.time()
    window_start = now - 60.0
    bucket = _RATE_LIMIT_BUCKET.setdefault(key, [])
    # prune old entries
    i = 0
    for i in range(len(bucket)):
        if bucket[i] >= window_start:
            break
    if i > 0:
        del bucket[:i]
    if len(bucket) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)
    return True

# Include routers with security dependencies
secured_deps = [Depends(verify_api_key), Depends(rate_limit)]
app.include_router(pipelines_router, prefix="/api/pipelines", tags=["pipelines"], dependencies=secured_deps)
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["monitoring"], dependencies=secured_deps)
app.include_router(models_router, prefix="/api/models", tags=["models"], dependencies=secured_deps)
app.include_router(config_router, prefix="/api/config", tags=["configuration"], dependencies=secured_deps)

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
