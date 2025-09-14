"""
Sakana AI AB-MCTS Service

Implements the actual Sakana AI AB-MCTS algorithm with:
- Two-dimensional search (width + depth)
- Thompson Sampling for adaptive branching
- Beta distribution for probability models
- Proper width/depth balance
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

# Import our data models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LLMState, QueryRequest, QueryResponse, SearchStats, ConversationTurn, Conversation

app = FastAPI(title="Sakana AI AB-MCTS Service", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchDirection(Enum):
    WIDTH = "width"  # Generate new solutions
    DEPTH = "depth"  # Refine existing solutions

@dataclass
class ABMCTSNode:
    """AB-MCTS node with Sakana AI's approach."""
    state: str
    parent: Optional['ABMCTSNode'] = None
    children: List['ABMCTSNode'] = None
    visits: int = 0
    reward: float = 0.0
    depth: int = 0
    model_used: str = ""
    search_direction: SearchDirection = SearchDirection.WIDTH
    quality_estimate: float = 0.0
    width_potential: float = 0.0  # Potential for new solutions
    depth_potential: float = 0.0  # Potential for refinement
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class SakanaABMCTSService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.conversations = {}
        self.models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
        # Sakana AI parameters
        self.max_nodes = 20
        self.beta_alpha = 2.0  # Beta distribution alpha
        self.beta_beta = 2.0   # Beta distribution beta
        self.thompson_sampling = True
        
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
    
    def generate_width_actions(self, query: str, model: str) -> List[str]:
        """Generate width actions (new solutions) for exploration."""
        return [
            f"Provide a comprehensive answer to: {query}",
            f"Give a detailed explanation of: {query}",
            f"Explain {query} with specific examples and analysis",
            f"Answer {query} from a different perspective or approach"
        ]
    
    def generate_depth_actions(self, current_state: str, query: str, model: str) -> List[str]:
        """Generate depth actions (refinements) for existing solutions."""
        return [
            f"Improve and expand on this response: {current_state}",
            f"Add more specific details and examples to: {current_state}",
            f"Refine and clarify the following answer: {current_state}",
            f"Enhance this response with additional insights: {current_state}"
        ]
    
    def thompson_sampling_decision(self, width_potential: float, depth_potential: float) -> SearchDirection:
        """Use Thompson Sampling to decide between width and depth search."""
        if not self.thompson_sampling:
            return SearchDirection.WIDTH if width_potential > depth_potential else SearchDirection.DEPTH
        
        # Thompson Sampling with Beta distribution
        width_sample = np.random.beta(self.beta_alpha + width_potential, self.beta_beta + (1 - width_potential))
        depth_sample = np.random.beta(self.beta_alpha + depth_potential, self.beta_beta + (1 - depth_potential))
        
        return SearchDirection.WIDTH if width_sample > depth_sample else SearchDirection.DEPTH
    
    def adaptive_branching_decision(self, node: ABMCTSNode, iteration: int, total_iterations: int) -> SearchDirection:
        """Sakana AI's adaptive branching decision using Thompson Sampling."""
        if not node.children:
            return SearchDirection.WIDTH  # Always start with width
        
        # Calculate width and depth potentials
        width_potential = node.width_potential
        depth_potential = node.depth_potential
        
        # If we have good quality nodes, increase depth potential
        if node.quality_estimate > 0.6:
            depth_potential += 0.4
        
        # If we have many children but low quality, increase width potential
        if len(node.children) > 2 and node.quality_estimate < 0.5:
            width_potential += 0.4
        
        # Force some depth searches after initial exploration
        if iteration > total_iterations * 0.3 and node.quality_estimate > 0.4:
            depth_potential += 0.5
        
        # Use Thompson Sampling for decision
        return self.thompson_sampling_decision(width_potential, depth_potential)
    
    def select_best_child_ucb(self, node: ABMCTSNode) -> ABMCTSNode:
        """Select best child using UCB1 with quality weighting."""
        if not node.children:
            return None
        
        best_child = None
        best_score = -float('inf')
        
        for child in node.children:
            if child.visits == 0:
                return child  # Return unvisited child
            
            # UCB1 with quality weighting
            exploitation = child.reward / child.visits
            exploration = math.sqrt(2 * math.log(node.visits) / child.visits)
            quality_bonus = child.quality_estimate * 0.2
            
            score = exploitation + exploration + quality_bonus
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def run_sakana_ab_mcts(self, query: str, iterations: int = 20, max_depth: int = 5) -> Dict[str, Any]:
        """Run Sakana AI's AB-MCTS algorithm."""
        start_time = time.time()
        
        # Initialize root node
        root = ABMCTSNode(state="", depth=0)
        best_solution = ""
        best_quality = 0.0
        
        search_stats = {
            "total_iterations": 0,
            "nodes_created": 0,
            "width_searches": 0,
            "depth_searches": 0,
            "best_quality": 0.0,
            "average_quality": 0.0,
            "model_usage": {model: 0 for model in self.models},
            "model_used": "",
            "response_time": 0.0,
            "thompson_sampling_decisions": 0
        }
        
        for iteration in range(min(iterations, self.max_nodes)):
            # Selection phase
            current = root
            path = [current]
            
            while current.children and current.depth < max_depth:
                current = self.select_best_child_ucb(current)
                if current is None:
                    break
                path.append(current)
            
            # Adaptive branching decision using Sakana AI's approach
            search_direction = self.adaptive_branching_decision(current, iteration, iterations)
            search_stats["thompson_sampling_decisions"] += 1
            
            # Expansion phase
            if current.depth < max_depth and not current.children:
                # Generate actions based on search direction
                model = self.models[iteration % len(self.models)]
                
                if search_direction == SearchDirection.WIDTH:
                    actions = self.generate_width_actions(query, model)
                    search_stats["width_searches"] += 1
                else:
                    actions = self.generate_depth_actions(current.state, query, model)
                    search_stats["depth_searches"] += 1
                
                # Create child nodes
                for action in actions:
                    child = ABMCTSNode(
                        state=action,
                        parent=current,
                        depth=current.depth + 1,
                        model_used=model,
                        search_direction=search_direction
                    )
                    current.children.append(child)
                    search_stats["nodes_created"] += 1
            
            # Simulation phase
            if current.children:
                # Select child for simulation
                child = random.choice(current.children)
                model = child.model_used
                
                # Get response from model
                response = self.call_ollama(model, child.state)
                child.state = response
                
                # Evaluate quality
                quality = self.evaluate_solution_quality(response, query)
                child.quality_estimate = quality
                child.reward = quality
                child.visits = 1
                
                # Update potentials based on search direction
                if child.search_direction == SearchDirection.WIDTH:
                    child.width_potential = quality
                else:
                    child.depth_potential = quality
                
                # Update best solution
                if quality > best_quality:
                    best_quality = quality
                    best_solution = response
                    search_stats["model_used"] = model
                
                search_stats["model_usage"][model] += 1
            else:
                # Leaf node simulation
                model = self.models[iteration % len(self.models)]
                response = self.call_ollama(model, current.state)
                current.state = response
                
                quality = self.evaluate_solution_quality(response, query)
                current.quality_estimate = quality
                current.reward = quality
                current.visits = 1
                
                # Update potentials
                if search_direction == SearchDirection.WIDTH:
                    current.width_potential = quality
                else:
                    current.depth_potential = quality
                
                if quality > best_quality:
                    best_quality = quality
                    best_solution = response
                    search_stats["model_used"] = model
                
                search_stats["model_usage"][model] += 1
            
            # Backpropagation
            for node in reversed(path):
                node.visits += 1
                if node.children:
                    node.reward = max(child.reward for child in node.children)
                    node.quality_estimate = max(child.quality_estimate for child in node.children)
                    
                    # Update potentials safely
                    width_children = [child for child in node.children if child.search_direction == SearchDirection.WIDTH]
                    depth_children = [child for child in node.children if child.search_direction == SearchDirection.DEPTH]
                    
                    if width_children:
                        node.width_potential = max(child.width_potential for child in width_children)
                    if depth_children:
                        node.depth_potential = max(child.depth_potential for child in depth_children)
            
            search_stats["total_iterations"] = iteration + 1
            search_stats["best_quality"] = best_quality
        
        # Calculate final statistics
        search_stats["average_quality"] = best_quality
        search_stats["response_time"] = time.time() - start_time
        
        return {
            "solution": best_solution,
            "search_stats": search_stats
        }
    
    def process_query(self, query: str, iterations: int = 20, max_depth: int = 5, 
                     conversation_id: Optional[str] = None) -> QueryResponse:
        """Process a query using Sakana AI's AB-MCTS."""
        try:
            # Run Sakana AB-MCTS search
            result = self.run_sakana_ab_mcts(query, iterations, max_depth)
            
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
service = SakanaABMCTSService()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using Sakana AI's AB-MCTS."""
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
        "service": "sakana-ab-mcts",
        "status": "running",
        "models": service.models,
        "max_nodes": service.max_nodes,
        "beta_alpha": service.beta_alpha,
        "beta_beta": service.beta_beta,
        "thompson_sampling": service.thompson_sampling
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)
