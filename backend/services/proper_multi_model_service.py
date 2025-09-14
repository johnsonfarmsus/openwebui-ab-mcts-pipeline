"""
Proper Multi-Model Service

Implements sophisticated multi-model collaboration with:
- Model-specific prompting strategies
- Advanced response synthesis
- Quality-based model selection
- Iterative refinement
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

app = FastAPI(title="Proper Multi-Model Service", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProperMultiModelService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.models = {
            "deepseek-r1:1.5b": {
                "name": "DeepSeek-R1",
                "strength": "reasoning",
                "temperature": 0.7,
                "prompt_style": "analytical"
            },
            "gemma3:1b": {
                "name": "Gemma-3",
                "strength": "efficiency",
                "temperature": 0.8,
                "prompt_style": "concise"
            },
            "llama3.2:1b": {
                "name": "Llama-3.2",
                "strength": "creativity",
                "temperature": 0.9,
                "prompt_style": "creative"
            }
        }
        
    def call_ollama(self, model: str, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Call Ollama API with model-specific parameters."""
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
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "model": model,
                "response_time": 0.0
            }
    
    def create_model_specific_prompt(self, query: str, model: str) -> str:
        """Create model-specific prompts based on each model's strengths."""
        model_info = self.models.get(model, {})
        prompt_style = model_info.get("prompt_style", "standard")
        strength = model_info.get("strength", "general")
        
        if prompt_style == "analytical":
            return f"""You are an analytical AI assistant. Please provide a thorough, step-by-step analysis of the following question:

Question: {query}

Please:
1. Break down the question into key components
2. Provide detailed reasoning for each part
3. Give specific examples where relevant
4. Conclude with a clear, well-structured answer

Analysis:"""
        
        elif prompt_style == "concise":
            return f"""You are an efficient AI assistant. Please provide a clear, concise answer to:

{query}

Focus on:
- Key points only
- Clear structure
- Practical insights
- Direct answers

Answer:"""
        
        elif prompt_style == "creative":
            return f"""You are a creative AI assistant. Please provide an engaging, creative response to:

{query}

Approach this with:
- Creative thinking
- Multiple perspectives
- Engaging examples
- Original insights

Response:"""
        
        else:
            return f"Please provide a comprehensive answer to: {query}"
    
    def evaluate_response_quality(self, response: str, query: str) -> float:
        """Evaluate the quality of a model response."""
        if not response or "Error:" in response:
            return 0.0
        
        # Length appropriateness (not too short, not too long)
        length_score = 1.0 - abs(len(response) - 500) / 1000
        length_score = max(0.0, min(1.0, length_score))
        
        # Relevance to query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        
        # Structure quality
        structure_indicators = ["1.", "2.", "3.", "First", "Second", "Third", "Therefore", "However", "In conclusion"]
        structure_score = sum(1 for indicator in structure_indicators if indicator in response) / len(structure_indicators)
        
        # Confidence indicators
        confidence_indicators = ["definitely", "certainly", "clearly", "specifically"]
        uncertainty_indicators = ["maybe", "perhaps", "might", "unclear", "not sure"]
        
        confidence_boost = sum(1 for indicator in confidence_indicators if indicator in response.lower()) * 0.1
        confidence_penalty = sum(1 for indicator in uncertainty_indicators if indicator in response.lower()) * 0.1
        
        # Calculate overall quality
        quality = (length_score * 0.2 + relevance_score * 0.4 + structure_score * 0.3 + 
                  (0.5 + confidence_boost - confidence_penalty) * 0.1)
        
        return max(0.0, min(1.0, quality))
    
    def synthesize_responses(self, responses: List[Dict[str, Any]], query: str) -> str:
        """Synthesize multiple model responses using advanced techniques."""
        successful_responses = [r for r in responses if r["success"]]
        
        if not successful_responses:
            return "I apologize, but I was unable to get responses from any of the models. Please check that Ollama is running and the models are available."
        
        if len(successful_responses) == 1:
            return successful_responses[0]["response"]
        
        # Evaluate quality of each response
        for resp in successful_responses:
            resp["quality"] = self.evaluate_response_quality(resp["response"], query)
        
        # Sort by quality
        successful_responses.sort(key=lambda x: x["quality"], reverse=True)
        
        # Use the best model for synthesis
        best_model = successful_responses[0]["model"]
        best_model_info = self.models.get(best_model, {})
        
        synthesis_prompt = f"""You are an expert synthesizer with {best_model_info.get('strength', 'general')} capabilities. 

I have multiple AI responses to the question: "{query}"

Here are the responses from different AI models:

"""
        
        for i, resp in enumerate(successful_responses, 1):
            model_name = self.models.get(resp["model"], {}).get("name", resp["model"])
            synthesis_prompt += f"Response {i} (from {model_name}, quality: {resp['quality']:.2f}):\n{resp['response']}\n\n"
        
        synthesis_prompt += """Please synthesize these responses into a single, comprehensive answer that:
1. Combines the best insights from each response
2. Resolves any contradictions or conflicts
3. Provides a clear, well-structured answer
4. Acknowledges different perspectives when appropriate
5. Maintains the highest quality information from each source

Synthesized Answer:"""
        
        # Get synthesis from best model
        synthesis_result = self.call_ollama(
            best_model, 
            synthesis_prompt, 
            temperature=0.6  # Lower temperature for synthesis
        )
        
        if synthesis_result["success"]:
            return synthesis_result["response"]
        else:
            # Fallback to weighted combination
            weighted_responses = []
            for resp in successful_responses:
                weight = resp["quality"]
                weighted_responses.append(f"**{self.models.get(resp['model'], {}).get('name', resp['model'])} (Quality: {weight:.2f}):** {resp['response']}")
            
            return "\n\n".join(weighted_responses)
    
    def process_query(self, query: str, models: Optional[List[str]] = None) -> QueryResponse:
        """Process a query using sophisticated multi-model collaboration."""
        start_time = time.time()
        
        # Use provided models or all available
        models_to_use = models or list(self.models.keys())
        
        # Call all models with model-specific prompts
        responses = []
        for model in models_to_use:
            prompt = self.create_model_specific_prompt(query, model)
            model_info = self.models.get(model, {})
            temperature = model_info.get("temperature", 0.7)
            
            response = self.call_ollama(model, prompt, temperature)
            responses.append(response)
        
        # Synthesize responses
        synthesized_response = self.synthesize_responses(responses, query)
        
        # Calculate statistics
        successful_responses = [r for r in responses if r["success"]]
        total_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_scores = [self.evaluate_response_quality(r["response"], query) for r in successful_responses]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        search_stats = {
            "total_iterations": len(models_to_use),
            "nodes_created": len(successful_responses),
            "best_reward": max(quality_scores) if quality_scores else 0.0,
            "average_reward": avg_quality,
            "exploration_ratio": 0.0,  # Not applicable for multi-model
            "width_searches": len(models_to_use),
            "depth_searches": 0,
            "model_usage": {model: 1 for model in models_to_use},
            "model_used": successful_responses[0]["model"] if successful_responses else "none",
            "response_time": total_time,
            "success_rate": len(successful_responses) / len(models_to_use),
            "quality_scores": {resp["model"]: self.evaluate_response_quality(resp["response"], query) 
                             for resp in successful_responses}
        }
        
        return QueryResponse(
            result=synthesized_response,
            success=len(successful_responses) > 0,
            search_stats=search_stats,
            conversation_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4())
        )

# Initialize service
service = ProperMultiModelService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using sophisticated multi-model collaboration."""
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
        "service": "proper-multi-model",
        "status": "running",
        "models": list(service.models.keys()),
        "model_info": service.models
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
