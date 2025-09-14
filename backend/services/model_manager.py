"""
Model Management Service

Handles model configuration, testing, performance tracking, and A/B testing.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
from dataclasses import dataclass, asdict
import uuid

from backend.models.model_config import ModelConfig
from backend.models.search_stats import SearchStats


@dataclass
class ModelTestResult:
    """Result of a model test."""
    model_id: str
    test_query: str
    response: str
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    variants: List[Dict[str, Any]]
    traffic_percentage: float
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, paused


class ModelManager:
    """Manages AI models, testing, and performance tracking."""
    
    def __init__(self, ollama_url: str = "http://host.docker.internal:11434"):
        self.ollama_url = ollama_url
        self.models: Dict[str, ModelConfig] = {}
        self.test_results: List[ModelTestResult] = []
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize with default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default models."""
        default_models = [
            {
                "id": "deepseek-r1-1.5b",
                "name": "DeepSeek-R1 1.5B",
                "endpoint": self.ollama_url,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9
                }
            },
            {
                "id": "gemma3-1b",
                "name": "Gemma3 1B",
                "endpoint": self.ollama_url,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9
                }
            },
            {
                "id": "llama3.2-1b",
                "name": "Llama3.2 1B",
                "endpoint": self.ollama_url,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9
                }
            }
        ]
        
        for model_data in default_models:
            model = ModelConfig.from_dict(model_data)
            self.models[model.id] = model
    
    async def test_model(self, model_id: str, test_query: str = "Hello, how are you?") -> ModelTestResult:
        """Test a model with a sample query."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        start_time = time.time()
        
        try:
            # Make actual API call to Ollama
            response = requests.post(
                f"{model.endpoint}/api/generate",
                json={
                    "model": model_id,
                    "prompt": test_query,
                    "stream": False,
                    **model.parameters
                },
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            response_text = response_data.get("response", "")
            response_time = time.time() - start_time
            
            # Update model usage
            model.update_usage(success=True)
            
            result = ModelTestResult(
                model_id=model_id,
                test_query=test_query,
                response=response_text,
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            model.update_usage(success=False)
            
            result = ModelTestResult(
                model_id=model_id,
                test_query=test_query,
                response="",
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    async def test_all_models(self, test_query: str = "Hello, how are you?") -> List[ModelTestResult]:
        """Test all enabled models with the same query."""
        enabled_models = [model for model in self.models.values() if model.enabled]
        results = []
        
        for model in enabled_models:
            result = await self.test_model(model.id, test_query)
            results.append(result)
        
        return results
    
    def get_model_performance(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter test results by time and model
        recent_tests = [
            result for result in self.test_results
            if result.model_id == model_id and 
            datetime.fromisoformat(result.timestamp) >= cutoff_time
        ]
        
        if not recent_tests:
            return {
                "model_id": model_id,
                "total_tests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "error_rate": 0.0
            }
        
        successful_tests = [t for t in recent_tests if t.success]
        response_times = [t.response_time for t in successful_tests]
        
        return {
            "model_id": model_id,
            "total_tests": len(recent_tests),
            "success_rate": len(successful_tests) / len(recent_tests),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "error_rate": (len(recent_tests) - len(successful_tests)) / len(recent_tests),
            "last_test": recent_tests[-1].timestamp if recent_tests else None
        }
    
    def get_all_models_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        performance = {}
        
        for model_id in self.models:
            performance[model_id] = self.get_model_performance(model_id, hours)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "models": performance
        }
    
    def create_ab_test(self, name: str, variants: List[Dict[str, Any]], 
                      traffic_percentage: float = 100.0, duration_hours: int = 24) -> str:
        """Create a new A/B test."""
        test_id = str(uuid.uuid4())
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        ab_test = ABTestConfig(
            test_id=test_id,
            name=name,
            variants=variants,
            traffic_percentage=traffic_percentage,
            start_time=datetime.now().isoformat(),
            end_time=end_time.isoformat()
        )
        
        self.ab_tests[test_id] = ab_test
        return test_id
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results for an A/B test."""
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        ab_test = self.ab_tests[test_id]
        
        # In a real implementation, you'd track actual usage data
        # For now, we'll return mock data
        return {
            "test_id": test_id,
            "name": ab_test.name,
            "status": ab_test.status,
            "start_time": ab_test.start_time,
            "end_time": ab_test.end_time,
            "variants": [
                {
                    "name": variant["name"],
                    "requests": 100,  # Mock data
                    "success_rate": 0.95,  # Mock data
                    "avg_response_time": 2.5  # Mock data
                }
                for variant in ab_test.variants
            ]
        }
    
    def get_model_recommendations(self) -> List[Dict[str, Any]]:
        """Get model recommendations based on performance."""
        recommendations = []
        
        for model_id, model in self.models.items():
            if not model.enabled:
                continue
            
            performance = self.get_model_performance(model_id, hours=24)
            
            # Simple recommendation logic
            if performance["success_rate"] > 0.9 and performance["avg_response_time"] < 3.0:
                recommendations.append({
                    "model_id": model_id,
                    "name": model.name,
                    "reason": "High success rate and fast response time",
                    "priority": "high"
                })
            elif performance["success_rate"] < 0.8:
                recommendations.append({
                    "model_id": model_id,
                    "name": model.name,
                    "reason": "Low success rate - consider investigation",
                    "priority": "medium"
                })
        
        return sorted(recommendations, key=lambda x: x["priority"], reverse=True)
    
    def export_model_config(self, model_id: str) -> Dict[str, Any]:
        """Export model configuration for backup/import."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        return {
            "export_timestamp": datetime.now().isoformat(),
            "model": model.to_dict(),
            "performance": self.get_model_performance(model_id)
        }
    
    def import_model_config(self, config_data: Dict[str, Any]) -> str:
        """Import model configuration from backup."""
        model_data = config_data["model"]
        model = ModelConfig.from_dict(model_data)
        
        # Generate new ID to avoid conflicts
        model.id = str(uuid.uuid4())
        
        self.models[model.id] = model
        return model.id
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all models."""
        total_models = len(self.models)
        enabled_models = len([m for m in self.models.values() if m.enabled])
        
        # Test each enabled model
        health_checks = {}
        for model_id, model in self.models.items():
            if not model.enabled:
                continue
            
            try:
                # Quick health check
                response = requests.get(f"{model.endpoint}/api/tags", timeout=5)
                health_checks[model_id] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                health_checks[model_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        healthy_models = len([h for h in health_checks.values() if h["status"] == "healthy"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_models": total_models,
            "enabled_models": enabled_models,
            "healthy_models": healthy_models,
            "health_percentage": (healthy_models / enabled_models * 100) if enabled_models > 0 else 0,
            "model_health": health_checks
        }
