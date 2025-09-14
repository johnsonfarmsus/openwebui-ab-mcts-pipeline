"""
Monitoring and analytics endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import json
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.monitoring_service import MonitoringService

router = APIRouter()

# Initialize monitoring service
monitoring_service = MonitoringService()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@router.get("/performance")
async def get_performance_metrics(
    hours: int = Query(24, description="Hours of data to retrieve"),
    service: Optional[str] = Query(None, description="Filter by service")
):
    """Get performance metrics."""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    # Filter data by time and service
    filtered_data = [
        entry for entry in performance_data
        if entry["timestamp"] >= cutoff_time and (service is None or entry.get("service") == service)
    ]
    
    if not filtered_data:
        return {
            "message": "No performance data available",
            "period_hours": hours,
            "service_filter": service
        }
    
    # Calculate metrics
    total_requests = len(filtered_data)
    successful_requests = len([entry for entry in filtered_data if entry.get("success", False)])
    success_rate = successful_requests / total_requests if total_requests > 0 else 0
    
    response_times = [entry.get("response_time", 0) for entry in filtered_data if entry.get("response_time")]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Group by service
    service_stats = {}
    for entry in filtered_data:
        svc = entry.get("service", "unknown")
        if svc not in service_stats:
            service_stats[svc] = {"requests": 0, "successful": 0, "response_times": []}
        
        service_stats[svc]["requests"] += 1
        if entry.get("success", False):
            service_stats[svc]["successful"] += 1
        if entry.get("response_time"):
            service_stats[svc]["response_times"].append(entry["response_time"])
    
    # Calculate service-specific metrics
    for svc, stats in service_stats.items():
        stats["success_rate"] = stats["successful"] / stats["requests"] if stats["requests"] > 0 else 0
        stats["avg_response_time"] = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
        del stats["response_times"]  # Remove raw data
    
    return {
        "period_hours": hours,
        "total_requests": total_requests,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "services": service_stats,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by log level"),
    service: Optional[str] = Query(None, description="Filter by service"),
    limit: int = Query(100, ge=1, le=1000, description="Number of logs to return")
):
    """Get system logs."""
    filtered_logs = logs_data.copy()
    
    # Apply filters
    if level:
        filtered_logs = [log for log in filtered_logs if log.get("level") == level]
    if service:
        filtered_logs = [log for log in filtered_logs if log.get("service") == service]
    
    # Sort by timestamp (newest first) and limit
    filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    filtered_logs = filtered_logs[:limit]
    
    return {
        "logs": filtered_logs,
        "total": len(filtered_logs),
        "filters": {
            "level": level,
            "service": service,
            "limit": limit
        },
        "timestamp": datetime.now().isoformat()
    }

@router.post("/logs")
async def add_log(log_data: Dict[str, Any]):
    """Add a log entry (for testing/demo purposes)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": log_data.get("level", "info"),
        "service": log_data.get("service", "unknown"),
        "message": log_data.get("message", ""),
        "metadata": log_data.get("metadata", {})
    }
    logs_data.append(log_entry)
    
    # Keep only last 1000 logs
    if len(logs_data) > 1000:
        logs_data[:] = logs_data[-1000:]
    
    return {"message": "Log entry added", "log_id": len(logs_data)}

@router.get("/metrics")
async def get_metrics():
    """Get detailed metrics for dashboard."""
    # This would typically query a metrics database
    return {
        "ab_mcts": {
            "total_queries": 150,
            "avg_score": 0.85,
            "avg_response_time": 3.2,
            "success_rate": 0.95,
            "models_used": {
                "deepseek-r1:1.5b": 60,
                "gemma3:1b": 45,
                "llama3.2:1b": 45
            }
        },
        "multi_model": {
            "total_queries": 200,
            "avg_confidence": 0.82,
            "avg_response_time": 1.8,
            "success_rate": 0.98,
            "models_used": {
                "deepseek-r1:1.5b": 70,
                "gemma3:1b": 65,
                "llama3.2:1b": 65
            }
        },
        "system": {
            "uptime": "2 days, 5 hours",
            "memory_usage": "4.2GB / 16GB",
            "cpu_usage": "15%",
            "disk_usage": "2.1GB / 100GB"
        }
    }

@router.get("/health")
async def get_health_status():
    """Get detailed health status of all services."""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "services": {}
    }
    
    # Check AB-MCTS service
    try:
        response = requests.get("http://localhost:8094/health", timeout=5)
        health_status["services"]["ab_mcts"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        health_status["services"]["ab_mcts"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }
        health_status["overall_status"] = "degraded"
    
    # Check Multi-Model service
    try:
        response = requests.get("http://localhost:8090/health", timeout=5)
        health_status["services"]["multi_model"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        health_status["services"]["multi_model"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }
        health_status["overall_status"] = "degraded"
    
    return health_status

# WebSocket endpoint for real-time monitoring
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic health updates
            health_data = await get_health_status()
            await manager.send_personal_message(
                json.dumps({
                    "type": "health_update",
                    "data": health_data
                }), 
                websocket
            )
            
            # Send performance metrics
            performance_data = monitoring_service.get_aggregated_metrics(hours=1)
            await manager.send_personal_message(
                json.dumps({
                    "type": "performance_update",
                    "data": performance_data
                }), 
                websocket
            )
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Enhanced monitoring endpoints

@router.get("/metrics/live")
async def get_live_metrics():
    """Get live metrics from monitoring service."""
    try:
        health_dashboard = monitoring_service.get_health_dashboard()
        return health_dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get live metrics: {str(e)}")

@router.get("/metrics/{metric_name}")
async def get_metric_data(metric_name: str, hours: int = 1):
    """Get data for a specific metric."""
    try:
        metrics = monitoring_service.get_metrics(metric_name, hours)
        return {
            "metric_name": metric_name,
            "period_hours": hours,
            "data_points": len(metrics),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metric data: {str(e)}")

@router.post("/metrics/record")
async def record_metric(metric_data: Dict[str, Any]):
    """Record a custom metric."""
    try:
        monitoring_service.record_metric(
            metric_name=metric_data["metric_name"],
            value=metric_data["value"],
            tags=metric_data.get("tags", {})
        )
        return {"message": "Metric recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.get("/alerts")
async def get_alerts():
    """Get all configured alerts."""
    try:
        alerts = monitoring_service.get_alerts()
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts")
async def create_alert(alert_data: Dict[str, Any]):
    """Create a new alert."""
    try:
        alert_id = monitoring_service.create_alert(
            name=alert_data["name"],
            condition=alert_data["condition"],
            threshold=alert_data["threshold"],
            severity=alert_data["severity"]
        )
        return {
            "message": "Alert created successfully",
            "alert_id": alert_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")

@router.get("/connections")
async def get_websocket_connections():
    """Get active WebSocket connections."""
    try:
        connections = monitoring_service.get_active_connections()
        return {
            "active_connections": len(manager.active_connections),
            "connections": connections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get connections: {str(e)}")
