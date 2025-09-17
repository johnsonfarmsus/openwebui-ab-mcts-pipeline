"""
Proper TreeQuest AB-MCTS Service

Uses the actual Sakana AI TreeQuest implementation with proper AB-MCTS algorithm.
Based on their official implementation and TreeQuest documentation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
import requests
import json
import uvicorn
import os
import time
import uuid
import math
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import TreeQuest
import treequest as tq

# Import our data models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LLMState, QueryRequest, QueryResponse, SearchStats, ConversationTurn, Conversation

# Import model discovery
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_discovery import ModelDiscoveryService

app = FastAPI(title="Proper TreeQuest AB-MCTS Service", version="6.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class NodeState:
    """Node state for AB-MCTS tree."""
    generation_result: str
    eval_results: Dict[str, Any]
    model_name: str
    search_type: str  # "width" or "depth"

class ProperTreeQuestABMCTSService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.conversations = {}
        
        # Initialize model discovery
        self.model_discovery = ModelDiscoveryService(self.ollama_url)
        
        # Discover available models and set defaults
        self.available_models = self.model_discovery.discover_models()
        self._persist_path = os.getenv("MODEL_SELECTION_FILE", "/app/logs/selected_models_abmcts.json")
        self.models = self._load_selected_models() or self.model_discovery.get_recommended_models(for_testing=True)
        
        # Sakana AI TreeQuest configuration
        self.algo_config = {
            "class_name": "ABMCTSA",
            "params": {
                "model_selection_strategy": "stack"
            }
        }
        
        # Initialize TreeQuest algorithm
        self.algo_cls = getattr(tq, self.algo_config["class_name"])
        self.algo = self.algo_cls(**self.algo_config["params"])
        
    def call_ollama(self, model: str, prompt: str, temperature: float = 0.6) -> str:
        """Call Ollama API with error handling."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "max_tokens": 1000
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response received")
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def evaluate_solution_quality(self, solution: str, query: str) -> float:
        """Evaluate solution quality using Sakana AI's approach."""
        if not solution or "Error:" in solution:
            return 0.0
        
        # Length-based quality
        length_score = min(len(solution) / 1000, 1.0)
        
        # Structure quality
        structure_indicators = ["1.", "2.", "3.", "First", "Second", "Third", "Therefore", "However", "In conclusion"]
        structure_score = sum(1 for indicator in structure_indicators if indicator in solution) / len(structure_indicators)
        
        # Relevance quality
        query_terms = set(query.lower().split())
        solution_terms = set(solution.lower().split())
        relevance_score = len(query_terms.intersection(solution_terms)) / max(len(query_terms), 1)
        
        # Confidence quality
        confidence_indicators = ["definitely", "certainly", "clearly", "specifically"]
        uncertainty_indicators = ["maybe", "perhaps", "might", "unclear", "not sure"]
        
        confidence_boost = sum(1 for indicator in confidence_indicators if indicator in solution.lower()) * 0.1
        confidence_penalty = sum(1 for indicator in uncertainty_indicators if indicator in solution.lower()) * 0.1
        
        # Calculate overall quality
        quality = (length_score * 0.3 + structure_score * 0.3 + relevance_score * 0.4) + confidence_boost - confidence_penalty
        
        return max(0.0, min(1.0, quality))
    
    def generate_width_prompt(self, query: str) -> str:
        """Generate width prompt for new solutions."""
        return f"""You are an expert AI assistant. Please provide a comprehensive answer to the following question:

Question: {query}

Please provide a detailed, well-structured response that:
1. Directly addresses the question
2. Provides specific examples where relevant
3. Explains key concepts clearly
4. Offers practical insights

Answer:"""
    
    def generate_depth_prompt(self, current_response: str, query: str) -> str:
        """Generate depth prompt for refining existing solutions."""
        return f"""You are an expert AI assistant. Please improve and expand on the following response:

Original Question: {query}

Current Response: {current_response}

Please enhance this response by:
1. Adding more specific details and examples
2. Clarifying any unclear points
3. Providing additional insights
4. Improving the structure and flow

Enhanced Response:"""
    
    def generate_fn(self, state: Optional[NodeState], model_name: str, query: str) -> Tuple[NodeState, float]:
        """Generate function for TreeQuest AB-MCTS."""
        start_time = time.time()
        
        # Determine if this is width or depth search
        if state is None or not state.generation_result:
            # Width search - generate new solution
            prompt = self.generate_width_prompt(query)
            search_type = "width"
        else:
            # Depth search - refine existing solution
            prompt = self.generate_depth_prompt(state.generation_result, query)
            search_type = "depth"
        
        # Get response from model
        response = self.call_ollama(model_name, prompt, temperature=0.6)
        
        # Evaluate quality
        quality = self.evaluate_solution_quality(response, query)
        
        # Create node state
        node_state = NodeState(
            generation_result=response,
            eval_results={"quality": quality, "search_type": search_type},
            model_name=model_name,
            search_type=search_type
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return node_state, quality
    
    def run_proper_treequest_ab_mcts(self, query: str, iterations: int = 20, max_depth: int = 5, include_tree: bool = False) -> Dict[str, Any]:
        """Run proper TreeQuest AB-MCTS algorithm."""
        start_time = time.time()
        
        # Create generate functions for each model
        generate_fns = {
            model: lambda state, model_name=model: self.generate_fn(state, model_name, query)
            for model in self.models
        }
        
        # Initialize search tree
        search_tree = self.algo.init_tree()
        
        # Run AB-MCTS iterations
        search_stats = {
            "total_iterations": 0,
            "nodes_created": 0,
            "width_searches": 0,
            "depth_searches": 0,
            "best_quality": 0.0,
            "average_quality": 0.0,
            "model_usage": {model: 0 for model in self.models},
            "model_used": "",
            "response_time": 0.0
        }
        
        iteration_log: List[Dict[str, Any]] = []
        for i in range(min(iterations, 20)):  # Limit to 20 iterations
            # Run one step of AB-MCTS
            search_tree = self.algo.step(search_tree, generate_fns)
            
            # Get current state-score pairs
            state_score_pairs = self.algo.get_state_score_pairs(search_tree)
            search_stats["nodes_created"] = len(state_score_pairs)
            if include_tree:
                # Compact snapshot for visualization
                iteration_log.append({
                    "iteration": i + 1,
                    "nodes": [
                        {
                            "model": getattr(state, 'model_name', getattr(state, 'model_used', 'unknown')),
                            "quality": score,
                            "search_type": getattr(state, 'search_type', getattr(state, 'eval_results', {}).get('search_type', 'unknown')),
                            "preview": (getattr(state, 'generation_result', getattr(state, 'content', '')) or '')[:120]
                        }
                        for state, score in state_score_pairs[:100]
                    ]
                })
            
            # Count search types
            width_count = 0
            depth_count = 0
            for state, score in state_score_pairs:
                if hasattr(state, 'search_type'):
                    if state.search_type == "width":
                        width_count += 1
                    elif state.search_type == "depth":
                        depth_count += 1
                elif hasattr(state, 'eval_results') and state.eval_results.get("search_type") == "width":
                    width_count += 1
                elif hasattr(state, 'eval_results') and state.eval_results.get("search_type") == "depth":
                    depth_count += 1
            
            search_stats["width_searches"] = width_count
            search_stats["depth_searches"] = depth_count
            
            # Update best quality
            if state_score_pairs:
                best_score = max(score for _, score in state_score_pairs)
                search_stats["best_quality"] = best_score
                
                # Find best model
                best_state = max(state_score_pairs, key=lambda x: x[1])[0]
                if hasattr(best_state, 'model_name'):
                    search_stats["model_used"] = best_state.model_name
            
            search_stats["total_iterations"] = i + 1
        
        # Get final best solution
        if state_score_pairs:
            best_state, best_score = max(state_score_pairs, key=lambda x: x[1])
            best_solution = best_state.generation_result if hasattr(best_state, 'generation_result') else str(best_state)
        else:
            best_solution = "No solution found"
            best_score = 0.0
        
        # Calculate final statistics
        search_stats["average_quality"] = best_score
        search_stats["response_time"] = time.time() - start_time
        
        return {
            "solution": best_solution,
            "search_stats": search_stats,
            "iteration_log": iteration_log if include_tree else []
        }
    
    def update_models(self, model_names: List[str]) -> bool:
        """Update the models used for AB-MCTS."""
        try:
            # Validate models are available
            validation = self.model_discovery.validate_models(model_names)
            unavailable = [name for name, available in validation.items() if not available]
            
            if unavailable:
                print(f"Warning: Some models not available: {unavailable}")
            
            # Update models list and persist
            self.models = [name for name in model_names if validation.get(name, False)]
            self._save_selected_models()
            
            if not self.models:
                print("Error: No valid models selected")
                return False
                
            print(f"Updated models: {self.models}")
            return True
            
        except Exception as e:
            print(f"Error updating models: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with metadata."""
        # Always refresh to reflect current Ollama availability
        self.available_models = self.model_discovery.discover_models()
        
        return [
            {
                "name": model.name,
                "size": model.size,
                "parameters": model.parameters,
                "speed_rating": model.speed_rating,
                "quality_rating": model.quality_rating,
                "recommended_for_testing": model.recommended_for_testing,
                "recommended_for_production": model.recommended_for_production,
                "currently_selected": model.name in self.models
            }
            for model in self.available_models
        ]

    # -------- Persistence helpers --------
    def _load_selected_models(self) -> List[str]:
        try:
            if os.path.exists(self._persist_path):
                import json
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("selected", [])
        except Exception:
            pass
        return []

    def _save_selected_models(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            import json
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump({"selected": self.models}, f)
        except Exception:
            pass
    
    def process_query(self, query: str, iterations: int = 20, max_depth: int = 5, 
                     conversation_id: Optional[str] = None, models: Optional[List[str]] = None) -> QueryResponse:
        """Process a query using proper TreeQuest AB-MCTS."""
        try:
            # Update models if provided
            if models:
                self.update_models(models)
            
            # Run proper TreeQuest AB-MCTS search
            include_tree = True  # always capture for research; UI can ignore
            result = self.run_proper_treequest_ab_mcts(query, iterations, max_depth, include_tree)
            
            # Create response
            response = QueryResponse(
                result=result["solution"],
                success=True,
                search_stats=result["search_stats"],
                conversation_id=conversation_id or str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                iteration_log=result.get("iteration_log", [])
            )
            
            return response
            
        except Exception as e:
            return QueryResponse(
                result="",
                success=False,
                search_stats={},
                conversation_id=conversation_id or str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                error=str(e)
            )

# Initialize service
service = ProperTreeQuestABMCTSService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using proper TreeQuest AB-MCTS."""
    try:
        response = service.process_query(
            query=request.query,
            iterations=request.iterations,
            max_depth=request.max_depth,
            conversation_id=request.conversation_id,
            models=getattr(request, 'models', None)
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available models with metadata."""
    try:
        models = service.get_available_models()
        return {
            "success": True,
            "models": models,
            "current_models": service.models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/update")
async def update_models(request: Dict[str, Any]):
    """Update the models used for AB-MCTS."""
    try:
        model_names = request.get("models", [])
        success = service.update_models(model_names)
        
        return {
            "success": success,
            "message": "Models updated successfully" if success else "Failed to update models",
            "current_models": service.models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/test")
async def test_models(request: Dict[str, Any]):
    """Test selected models with a simple query."""
    try:
        model_names = request.get("models", [])
        test_query = request.get("query", "What is 2+2?")
        
        # Temporarily update models
        original_models = service.models.copy()
        service.update_models(model_names)
        
        # Run test query
        response = service.process_query(
            query=test_query,
            iterations=5,
            max_depth=2
        )
        
        # Restore original models
        service.models = original_models
        
        return {
            "success": True,
            "test_query": test_query,
            "test_models": model_names,
            "response": response.result,
            "search_stats": response.search_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "service": "proper-treequest-ab-mcts",
        "status": "running",
        "models": service.models,
        "algo_config": service.algo_config
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)
