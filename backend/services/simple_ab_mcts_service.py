"""
Simplified AB-MCTS Service

A more robust implementation that focuses on core functionality
without complex TreeQuest dependencies that might cause issues.
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
from dataclasses import dataclass

# Import our data models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LLMState, QueryRequest, QueryResponse, SearchStats, ConversationTurn, Conversation

app = FastAPI(title="Simplified AB-MCTS Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class MCTSNode:
    """Simple MCTS node for tree search."""
    state: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    reward: float = 0.0
    depth: int = 0
    model_used: str = ""
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class SimpleABMCTSService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.conversations = {}
        self.models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
    def call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API with error handling."""
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
                return data.get("response", "No response received")
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def evaluate_solution(self, solution: str, query: str) -> float:
        """Simple solution evaluation."""
        if not solution or "Error:" in solution:
            return 0.0
        
        # Basic scoring based on length and content quality
        length_score = min(len(solution) / 500, 1.0)
        
        # Check for relevant content
        query_words = set(query.lower().split())
        solution_words = set(solution.lower().split())
        relevance_score = len(query_words.intersection(solution_words)) / max(len(query_words), 1)
        
        # Penalize error responses
        error_penalty = 0.0
        if "error" in solution.lower() or "sorry" in solution.lower():
            error_penalty = 0.3
        
        # Combine scores
        total_score = (length_score * 0.4 + relevance_score * 0.6) - error_penalty
        return max(0.0, min(1.0, total_score))
    
    def generate_actions(self, current_state: str, query: str, model: str) -> List[str]:
        """Generate possible actions for the current state."""
        if not current_state:
            # Initial state - generate direct responses
            return [
                f"Answer this question directly: {query}",
                f"Provide a comprehensive explanation for: {query}",
                f"Give a detailed response about: {query}"
            ]
        else:
            # Refinement state - improve existing response
            return [
                f"Improve and expand on this response: {current_state}",
                f"Add more details to: {current_state}",
                f"Refine and clarify: {current_state}"
            ]
    
    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB1."""
        if not node.children:
            return None
        
        import math
        best_child = None
        best_score = -float('inf')
        
        for child in node.children:
            if child.visits == 0:
                return child  # Return unvisited child
            
            # UCB1 formula
            exploitation = child.reward / child.visits
            exploration = math.sqrt(2 * math.log(node.visits) / child.visits)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def run_ab_mcts(self, query: str, iterations: int = 20, max_depth: int = 5) -> Dict[str, Any]:
        """Run AB-MCTS search."""
        start_time = time.time()
        
        # Initialize root node
        root = MCTSNode(state="", depth=0)
        best_solution = ""
        best_score = 0.0
        
        search_stats = {
            "total_iterations": 0,
            "nodes_created": 0,
            "best_reward": 0.0,
            "average_reward": 0.0,
            "exploration_ratio": 0.0,
            "width_searches": 0,
            "depth_searches": 0,
            "model_usage": {model: 0 for model in self.models},
            "model_used": ""
        }
        
        for iteration in range(iterations):
            # Selection phase
            current = root
            path = [current]
            
            while current.children and current.depth < max_depth:
                current = self.select_best_child(current)
                if current is None:
                    break
                path.append(current)
            
            # Expansion phase
            if current.depth < max_depth and not current.children:
                # Generate actions for current state
                model = self.models[iteration % len(self.models)]
                actions = self.generate_actions(current.state, query, model)
                
                # Create child nodes
                for action in actions:
                    child = MCTSNode(
                        state=action,
                        parent=current,
                        depth=current.depth + 1,
                        model_used=model
                    )
                    current.children.append(child)
                    search_stats["nodes_created"] += 1
                
                search_stats["width_searches"] += 1
            else:
                search_stats["depth_searches"] += 1
            
            # Simulation phase
            if current.children:
                # Select random child for simulation
                import random
                child = random.choice(current.children)
                model = child.model_used
                
                # Get response from model
                response = self.call_ollama(model, child.state)
                child.state = response
                
                # Evaluate response
                reward = self.evaluate_solution(response, query)
                child.reward = reward
                child.visits = 1
                
                # Update best solution
                if reward > best_score:
                    best_score = reward
                    best_solution = response
                    search_stats["model_used"] = model
                
                search_stats["model_usage"][model] += 1
            else:
                # Leaf node - simulate directly
                model = self.models[iteration % len(self.models)]
                response = self.call_ollama(model, current.state)
                current.state = response
                
                reward = self.evaluate_solution(response, query)
                current.reward = reward
                current.visits = 1
                
                if reward > best_score:
                    best_score = reward
                    best_solution = response
                    search_stats["model_used"] = model
                
                search_stats["model_usage"][model] += 1
            
            # Backpropagation
            for node in reversed(path):
                node.visits += 1
                if node.children:
                    node.reward = max(child.reward for child in node.children)
            
            search_stats["total_iterations"] = iteration + 1
            search_stats["best_reward"] = best_score
        
        # Calculate final statistics
        search_stats["average_reward"] = best_score
        search_stats["exploration_ratio"] = search_stats["depth_searches"] / max(search_stats["total_iterations"], 1)
        search_stats["response_time"] = time.time() - start_time
        
        return {
            "solution": best_solution,
            "search_stats": search_stats
        }
    
    def process_query(self, query: str, iterations: int = 20, max_depth: int = 5, 
                     conversation_id: Optional[str] = None) -> QueryResponse:
        """Process a query using AB-MCTS."""
        try:
            # Run AB-MCTS search
            result = self.run_ab_mcts(query, iterations, max_depth)
            
            # Create response
            response = QueryResponse(
                result=result["solution"],
                success=True,
                search_stats=result["search_stats"],
                conversation_id=conversation_id or str(uuid.uuid4()),
                turn_id=str(uuid.uuid4())
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
service = SimpleABMCTSService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using AB-MCTS."""
    try:
        response = service.process_query(
            query=request.query,
            iterations=request.iterations,
            max_depth=request.max_depth,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "service": "simple-ab-mcts",
        "status": "running",
        "models": service.models,
        "conversations": len(service.conversations)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)
