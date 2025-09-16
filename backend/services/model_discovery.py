"""
Model Discovery Service
Discovers available Ollama models and provides model selection functionality
"""

import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ModelSize(Enum):
    TINY = "tiny"      # < 1B parameters
    SMALL = "small"    # 1B-3B parameters  
    MEDIUM = "medium"  # 3B-7B parameters
    LARGE = "large"    # 7B+ parameters

@dataclass
class ModelInfo:
    name: str
    size: str
    parameters: str
    model_size: ModelSize
    speed_rating: int  # 1-5, 5 being fastest
    quality_rating: int  # 1-5, 5 being highest quality
    recommended_for_testing: bool
    recommended_for_production: bool

class ModelDiscoveryService:
    def __init__(self, ollama_base_url: str = "http://host.docker.internal:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_cache = {}
        self.last_discovery = None
        
    def discover_models(self) -> List[ModelInfo]:
        """Discover all available Ollama models with metadata"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            models = []
            
            for model in models_data.get('models', []):
                model_info = self._analyze_model(model['name'])
                models.append(model_info)
                
            # Sort by speed rating (fastest first) for testing
            models.sort(key=lambda x: (x.speed_rating, x.quality_rating), reverse=True)
            
            self.model_cache = {model.name: model for model in models}
            self.last_discovery = models
            
            return models
            
        except Exception as e:
            print(f"Error discovering models: {e}")
            return []
    
    def _analyze_model(self, model_name: str) -> ModelInfo:
        """Analyze model name to extract metadata"""
        name_lower = model_name.lower()
        
        # Extract parameter count
        if "135m" in name_lower or "0.1b" in name_lower:
            parameters = "135M"
            model_size = ModelSize.TINY
            speed_rating = 5
            quality_rating = 2
        elif "0.6b" in name_lower:
            parameters = "0.6B"
            model_size = ModelSize.TINY
            speed_rating = 4
            quality_rating = 3
        elif "1b" in name_lower or "1.5b" in name_lower:
            parameters = "1-1.5B"
            model_size = ModelSize.SMALL
            speed_rating = 4
            quality_rating = 3
        elif "3b" in name_lower:
            parameters = "3B"
            model_size = ModelSize.SMALL
            speed_rating = 3
            quality_rating = 4
        elif "7b" in name_lower:
            parameters = "7B"
            model_size = ModelSize.MEDIUM
            speed_rating = 2
            quality_rating = 4
        elif "13b" in name_lower:
            parameters = "13B"
            model_size = ModelSize.LARGE
            speed_rating = 1
            quality_rating = 5
        else:
            parameters = "Unknown"
            model_size = ModelSize.SMALL
            speed_rating = 3
            quality_rating = 3
        
        # Determine recommendations
        recommended_for_testing = model_size in [ModelSize.TINY, ModelSize.SMALL]
        recommended_for_production = model_size in [ModelSize.MEDIUM, ModelSize.LARGE]
        
        return ModelInfo(
            name=model_name,
            size=parameters,
            parameters=parameters,
            model_size=model_size,
            speed_rating=speed_rating,
            quality_rating=quality_rating,
            recommended_for_testing=recommended_for_testing,
            recommended_for_production=recommended_for_production
        )
    
    def get_recommended_models(self, for_testing: bool = True) -> List[str]:
        """Get recommended model names for testing or production"""
        if not self.model_cache:
            self.discover_models()
        
        if for_testing:
            return [name for name, model in self.model_cache.items() 
                   if model.recommended_for_testing]
        else:
            return [name for name, model in self.model_cache.items() 
                   if model.recommended_for_production]
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed info for a specific model"""
        if not self.model_cache:
            self.discover_models()
        return self.model_cache.get(model_name)
    
    def validate_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Validate that the requested models are available"""
        if not self.model_cache:
            self.discover_models()
        
        available_models = set(self.model_cache.keys())
        return {name: name in available_models for name in model_names}

# Global instance
model_discovery = ModelDiscoveryService()
