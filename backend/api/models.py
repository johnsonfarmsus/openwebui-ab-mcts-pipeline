"""
Model management endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.model_config import ModelConfig
from backend.services.model_manager import ModelManager

router = APIRouter()

# Initialize model manager
model_manager = ModelManager()

# Enhanced model management endpoints using ModelManager (single source of truth)

@router.get("/")
async def list_models():
    """List all available models."""
    models = [model.to_dict() for model in model_manager.models.values()]
    return {
        "models": models,
        "total": len(models),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get a specific model by ID."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_manager.models[model_id].to_dict()

@router.post("/")
async def create_model(model_data: Dict[str, Any]):
    """Create a new model."""
    if "id" not in model_data:
        model_data["id"] = str(uuid.uuid4())
    required_fields = ["name", "endpoint", "parameters"]
    for field in required_fields:
        if field not in model_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    if model_data["id"] in model_manager.models:
        raise HTTPException(status_code=409, detail="Model with this ID already exists")
    try:
        model = ModelConfig.from_dict(model_data)
        model_manager.models[model.id] = model
        return model.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid model data: {str(e)}")

@router.put("/{model_id}")
async def update_model(model_id: str, model_data: Dict[str, Any]):
    """Update an existing model."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    existing_model = model_manager.models[model_id]
    for field, value in model_data.items():
        if hasattr(existing_model, field):
            setattr(existing_model, field, value)
    existing_model.performance_score = existing_model.calculate_performance_score()
    return existing_model.to_dict()

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    del model_manager.models[model_id]
    return {"message": "Model deleted successfully"}

# (Removed duplicate test endpoint that simulated responses; using enhanced test below)

@router.get("/{model_id}/stats")
async def get_model_stats(model_id: str):
    """Get statistics for a specific model."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = model_manager.models[model_id]
    return {
        "model_id": model_id,
        "usage_count": model.usage_count,
        "error_count": model.error_count,
        "success_rate": model.calculate_performance_score(),
        "performance_score": model.performance_score,
        "last_used": model.last_used.isoformat() if model.last_used else None,
        "enabled": model.enabled,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/{model_id}/enable")
async def enable_model(model_id: str):
    """Enable a model."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_manager.models[model_id].enabled = True
    return {"message": "Model enabled successfully"}

@router.post("/{model_id}/disable")
async def disable_model(model_id: str):
    """Disable a model."""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_manager.models[model_id].enabled = False
    return {"message": "Model disabled successfully"}

@router.get("/enabled/list")
async def list_enabled_models():
    """List only enabled models."""
    enabled_models = [model for model in model_manager.models.values() if model.enabled]
    return {
        "models": [model.to_dict() for model in enabled_models],
        "total": len(enabled_models),
        "timestamp": datetime.now().isoformat()
    }

# Enhanced endpoints using ModelManager

@router.post("/{model_id}/test")
async def test_model_enhanced(model_id: str, test_data: Dict[str, Any]):
    """Test a model with enhanced testing capabilities."""
    test_query = test_data.get("query", "Hello, how are you?")
    
    try:
        result = await model_manager.test_model(model_id, test_query)
        return {
            "model_id": model_id,
            "test_query": test_query,
            "response": result.response,
            "response_time": result.response_time,
            "success": result.success,
            "error_message": result.error_message,
            "timestamp": result.timestamp
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test failed: {str(e)}")

@router.post("/test-all")
async def test_all_models(test_data: Dict[str, Any]):
    """Test all enabled models with the same query."""
    test_query = test_data.get("query", "Hello, how are you?")
    
    try:
        results = await model_manager.test_all_models(test_query)
        return {
            "test_query": test_query,
            "results": [
                {
                    "model_id": result.model_id,
                    "response": result.response,
                    "response_time": result.response_time,
                    "success": result.success,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp
                }
                for result in results
            ],
            "total_tested": len(results),
            "successful_tests": len([r for r in results if r.success])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch test failed: {str(e)}")

@router.get("/{model_id}/performance")
async def get_model_performance(model_id: str, hours: int = 24):
    """Get performance metrics for a specific model."""
    try:
        performance = model_manager.get_model_performance(model_id, hours)
        return performance
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/performance/all")
async def get_all_models_performance(hours: int = 24):
    """Get performance metrics for all models."""
    try:
        performance = model_manager.get_all_models_performance(hours)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance data: {str(e)}")

@router.get("/recommendations")
async def get_model_recommendations():
    """Get model recommendations based on performance."""
    try:
        recommendations = model_manager.get_model_recommendations()
        return {
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/health/status")
async def get_models_health_status():
    """Get health status of all models."""
    try:
        health_status = model_manager.get_health_status()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")

@router.post("/{model_id}/export")
async def export_model_config(model_id: str):
    """Export model configuration for backup."""
    try:
        config = model_manager.export_model_config(model_id)
        return config
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export model: {str(e)}")

@router.post("/import")
async def import_model_config(config_data: Dict[str, Any]):
    """Import model configuration from backup."""
    try:
        model_id = model_manager.import_model_config(config_data)
        return {
            "message": "Model imported successfully",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import model: {str(e)}")

@router.post("/ab-test")
async def create_ab_test(test_data: Dict[str, Any]):
    """Create a new A/B test."""
    try:
        test_id = model_manager.create_ab_test(
            name=test_data["name"],
            variants=test_data["variants"],
            traffic_percentage=test_data.get("traffic_percentage", 100.0),
            duration_hours=test_data.get("duration_hours", 24)
        )
        return {
            "message": "A/B test created successfully",
            "test_id": test_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create A/B test: {str(e)}")

@router.get("/ab-test/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get results for an A/B test."""
    try:
        results = model_manager.get_ab_test_results(test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get A/B test results: {str(e)}")
