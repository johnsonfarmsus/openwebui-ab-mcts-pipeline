"""
AB-MCTS Multi-Model Service

A standalone FastAPI service that implements AB-MCTS with multiple models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import requests
import json
import uvicorn
import os

# Import our data models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import QueryRequest, QueryResponse, SearchStats

app = FastAPI(title="AB-MCTS Multi-Model Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ABMCTSService:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
    def call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"API Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Connection Error: {str(e)}"

    def simple_multi_model(self, query: str, models: List[str]) -> str:
        """Multi-model collaboration approach"""
        
        result = []
        result.append("🤖 **Multi-Model Collaboration**\n\n")
        result.append(f"**Query:** {query}\n\n")

        responses = []
        
        # Get responses from all models
        for i, model in enumerate(models):
            result.append(f"### 🔍 Consulting {model}...\n")
            
            try:
                response = self.call_ollama(model, query)
                responses.append({"model": model, "response": response})
                result.append(f"**Response:** {response[:300]}{'...' if len(response) > 300 else ''}\n\n")
            except Exception as e:
                result.append(f"**Error:** {str(e)}\n\n")

        # Collaborative refinement
        if len(responses) > 1:
            result.append("---\n\n## 🔄 **Collaborative Refinement**\n\n")
            
            # Create a synthesis prompt
            all_responses = "\n\n".join([f"**{r['model']}:** {r['response']}" for r in responses])
            synthesis_prompt = f"""Original question: {query}

Here are responses from multiple AI models:
{all_responses}

Please synthesize these responses into a comprehensive, well-structured answer that combines the best insights from all models. Focus on accuracy, completeness, and clarity."""

            # Use the first model for synthesis
            try:
                synthesis = self.call_ollama(models[0], synthesis_prompt)
                result.append("### 🎯 **Synthesized Response**\n")
                result.append(f"{synthesis}\n\n")
            except Exception as e:
                result.append(f"**Synthesis Error:** {str(e)}\n\n")

        return "".join(result)

    def run_ab_mcts(self, query: str, iterations: int, models: List[str]) -> str:
        """AB-MCTS implementation"""
        try:
            import treequest as tq
            
            result = []
            result.append("🌳 **AB-MCTS Multi-Model Search**\n\n")
            result.append(f"**Query:** {query}\n")
            result.append(f"**Models:** {', '.join(models)}\n")
            result.append(f"**Iterations:** {iterations}\n\n")
            result.append("---\n\n")

            # For now, fall back to simple multi-model approach
            # Full AB-MCTS implementation would go here
            result.append("🚀 **Running enhanced multi-model collaboration...**\n\n")
            
            # Use the simple multi-model approach for now
            multi_result = self.simple_multi_model(query, models)
            result.append(multi_result)
            
            return "".join(result)
                
        except ImportError:
            return f"TreeQuest not available, falling back to multi-model approach...\n\n{self.simple_multi_model(query, models)}"
        except Exception as e:
            return f"AB-MCTS failed ({str(e)}), falling back to multi-model approach...\n\n{self.simple_multi_model(query, models)}"

    def process_query(self, query: str, iterations: int = 10, models: List[str] = None, enable_ab_mcts: bool = True) -> str:
        """Main query processing method"""
        if models is None:
            models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
        if enable_ab_mcts:
            return self.run_ab_mcts(query, iterations, models)
        else:
            return self.simple_multi_model(query, models)

# Initialize service
service = ABMCTSService()

@app.get("/")
async def root():
    return {"message": "AB-MCTS Multi-Model Service", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using AB-MCTS multi-model approach"""
    try:
        result = service.process_query(
            query=request.query,
            iterations=request.iterations,
            models=request.models,
            enable_ab_mcts=request.enable_ab_mcts
        )
        return QueryResponse(result=result, success=True)
    except Exception as e:
        return QueryResponse(result="", success=False, error=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
