"""
Simplified Multi-Model Service

A robust multi-model collaboration service that calls multiple models
and synthesizes their responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import requests
import json
import uvicorn
import os
import time
import uuid
from datetime import datetime

# Import our data models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import QueryRequest, QueryResponse, SearchStats

app = FastAPI(title="Simplified Multi-Model Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleMultiModelService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
    def call_ollama(self, model: str, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with comprehensive error handling."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 1000
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", "No response received"),
                    "model": model,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "model": model,
                    "response_time": response.elapsed.total_seconds()
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "model": model,
                "response_time": 30.0
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - Ollama not reachable",
                "model": model,
                "response_time": 0.0
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "model": model,
                "response_time": 0.0
            }
    
    def synthesize_responses(self, responses: List[Dict[str, Any]], query: str) -> str:
        """Synthesize multiple model responses into a single answer."""
        successful_responses = [r for r in responses if r["success"]]
        
        if not successful_responses:
            return "I apologize, but I was unable to get responses from any of the models. Please check that Ollama is running and the models are available."
        
        if len(successful_responses) == 1:
            return successful_responses[0]["response"]
        
        # Multiple successful responses - synthesize them
        synthesis_prompt = f"""You are an expert synthesizer. I have multiple responses to the question: "{query}"

Here are the responses from different AI models:

"""
        
        for i, resp in enumerate(successful_responses, 1):
            synthesis_prompt += f"Response {i} (from {resp['model']}):\n{resp['response']}\n\n"
        
        synthesis_prompt += """Please synthesize these responses into a single, comprehensive answer that:
1. Combines the best insights from each response
2. Resolves any contradictions
3. Provides a clear, well-structured answer
4. Acknowledges different perspectives when appropriate

Synthesized Answer:"""
        
        # Use the first successful model for synthesis
        synthesis_result = self.call_ollama(successful_responses[0]["model"], synthesis_prompt)
        
        if synthesis_result["success"]:
            return synthesis_result["response"]
        else:
            # Fallback to simple concatenation
            return "\n\n".join([f"**{resp['model']}:** {resp['response']}" for resp in successful_responses])
    
    def process_query(self, query: str, models: Optional[List[str]] = None) -> QueryResponse:
        """Process a query using multiple models."""
        start_time = time.time()
        
        # Use provided models or default
        models_to_use = models or self.models
        
        # Call all models in parallel
        responses = []
        for model in models_to_use:
            response = self.call_ollama(model, query)
            responses.append(response)
        
        # Synthesize responses
        synthesized_response = self.synthesize_responses(responses, query)
        
        # Calculate statistics
        successful_responses = [r for r in responses if r["success"]]
        total_time = time.time() - start_time
        
        search_stats = {
            "total_iterations": len(models_to_use),
            "nodes_created": len(successful_responses),
            "best_reward": 1.0 if successful_responses else 0.0,
            "average_reward": len(successful_responses) / len(models_to_use),
            "exploration_ratio": 0.0,  # Not applicable for multi-model
            "width_searches": len(models_to_use),
            "depth_searches": 0,
            "model_usage": {model: 1 for model in models_to_use},
            "model_used": successful_responses[0]["model"] if successful_responses else "none",
            "response_time": total_time,
            "success_rate": len(successful_responses) / len(models_to_use)
        }
        
        return QueryResponse(
            result=synthesized_response,
            success=len(successful_responses) > 0,
            search_stats=search_stats,
            conversation_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4())
        )

# Initialize service
service = SimpleMultiModelService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using multiple models."""
    try:
        response = service.process_query(
            query=request.query,
            models=request.models
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "service": "simple-multi-model",
        "status": "running",
        "models": service.models,
        "ollama_url": service.ollama_url
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
