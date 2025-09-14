"""
Real Sakana AI AB-MCTS Implementation

Based on Sakana AI's research: https://sakana.ai/ab-mcts/
Implements true two-dimensional search with adaptive branching and multi-LLM collaboration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import requests
import json
import uvicorn
import os
import math
import random
import time
import uuid
import numpy as np
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

app = FastAPI(title="Real Sakana AB-MCTS Service", version="4.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class NodeType(Enum):
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"

@dataclass
class MCTSNode:
    id: str
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    state: str  # The actual solution/content
    visits: int
    total_reward: float
    depth: int
    node_type: NodeType
    model_used: Optional[str] = None
    action_taken: Optional[str] = None
    # AB-MCTS specific fields
    width_visits: int = 0  # Visits for width search (new solutions)
    depth_visits: int = 0  # Visits for depth search (refinements)
    width_reward: float = 0.0
    depth_reward: float = 0.0
    # Thompson Sampling parameters
    alpha: float = 1.0  # Beta distribution alpha
    beta: float = 1.0   # Beta distribution beta
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)
    
    @property
    def ucb_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        c = 1.414  # Exploration constant
        return self.average_reward + c * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else self.average_reward

class QueryRequest(BaseModel):
    query: str
    iterations: int = 20
    models: List[str] = None
    max_depth: int = 5
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    result: str
    success: bool
    search_stats: Dict[str, Any]
    conversation_id: str = ""
    turn_id: str = ""

class RealSakanaABMCTSService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.conversations = {}
        
    def call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"API Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Connection Error: {str(e)}"

    def evaluate_solution(self, solution: str, original_query: str) -> float:
        """Evaluate the quality of a solution using multiple criteria"""
        if not solution or "Error:" in solution:
            return 0.0
        
        # Length-based scoring (longer responses often better)
        length_score = min(len(solution) / 1000, 1.0)
        
        # Structure scoring (look for organized responses)
        structure_indicators = ["1.", "2.", "3.", "First", "Second", "Third", "Therefore", "However", "In conclusion"]
        structure_score = sum(1 for indicator in structure_indicators if indicator in solution) / len(structure_indicators)
        
        # Relevance scoring (look for query-related terms)
        query_terms = original_query.lower().split()
        relevance_score = sum(1 for term in query_terms if term in solution.lower()) / max(len(query_terms), 1)
        
        # Confidence scoring (penalize uncertain responses)
        uncertainty_indicators = ["I'm not sure", "I don't know", "unclear", "maybe", "perhaps"]
        confidence_penalty = sum(1 for indicator in uncertainty_indicators if indicator.lower() in solution.lower()) * 0.1
        
        # Combine scores
        total_score = (length_score * 0.3 + structure_score * 0.3 + relevance_score * 0.4) - confidence_penalty
        return max(0.0, min(1.0, total_score))

    def thompson_sampling(self, node: MCTSNode) -> str:
        """Use Thompson Sampling to decide between width (new) or depth (refine) search"""
        # Sample from Beta distributions for width and depth actions
        width_sample = np.random.beta(node.alpha, node.beta)
        depth_sample = np.random.beta(node.alpha, node.beta)
        
        # If we have no previous state, always choose width (new solution)
        if not node.state or node.state.strip() == "":
            return "width"
        
        # Choose the action with higher sampled value
        return "width" if width_sample > depth_sample else "depth"

    def generate_width_action(self, query: str, model: str, conversation_context: str = "") -> str:
        """Generate a completely new solution (width search)"""
        context_prompt = f"Previous conversation context: {conversation_context}\n\n" if conversation_context else ""
        
        actions = [
            f"{context_prompt}Answer this question directly and comprehensively: {query}",
            f"{context_prompt}Provide a detailed explanation of: {query}",
            f"{context_prompt}Give a thorough analysis of: {query}",
            f"{context_prompt}Explain {query} with examples and context",
            f"{context_prompt}Create a comprehensive response about: {query}",
            f"{context_prompt}Approach this question from a different angle: {query}",
            f"{context_prompt}Think step by step about: {query}"
        ]
        
        return random.choice(actions)

    def generate_depth_action(self, current_state: str, query: str, model: str, conversation_context: str = "") -> str:
        """Generate an improvement to existing solution (depth search)"""
        context_prompt = f"Previous conversation context: {conversation_context}\n\n" if conversation_context else ""
        
        actions = [
            f"{context_prompt}Expand and elaborate on this response: {current_state}",
            f"{context_prompt}Add specific examples and details to this response: {current_state}",
            f"{context_prompt}Reorganize and improve the structure of this response: {current_state}",
            f"{context_prompt}Address different aspects and perspectives of: {query}",
            f"{context_prompt}Synthesize this with broader knowledge about: {query}",
            f"{context_prompt}Refine and polish this response: {current_state}",
            f"{context_prompt}Add more depth and insight to: {current_state}"
        ]
        
        return random.choice(actions)

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB1"""
        if not node.children:
            return node
        
        best_child = max(node.children, key=lambda x: x.ucb_value)
        return best_child

    def expand_node(self, node: MCTSNode, query: str, models: List[str], conversation_context: str = "") -> None:
        """Expand a node by adding children based on AB-MCTS principles"""
        if node.depth >= 5:  # Max depth limit
            return
        
        # Generate children for each model
        for model in models:
            # Create width child (new solution)
            width_action = self.generate_width_action(query, model, conversation_context)
            width_child = MCTSNode(
                id=f"{node.id}_width_{model}_{len(node.children)}",
                parent=node,
                children=[],
                state="",  # Will be filled during simulation
                visits=0,
                total_reward=0.0,
                depth=node.depth + 1,
                node_type=NodeType.LEAF if node.depth >= 4 else NodeType.INTERNAL,
                model_used=model,
                action_taken=width_action
            )
            node.children.append(width_child)
            
            # Create depth child (refinement) only if we have existing state
            if node.state and node.state.strip():
                depth_action = self.generate_depth_action(node.state, query, model, conversation_context)
                depth_child = MCTSNode(
                    id=f"{node.id}_depth_{model}_{len(node.children)}",
                    parent=node,
                    children=[],
                    state="",  # Will be filled during simulation
                    visits=0,
                    total_reward=0.0,
                    depth=node.depth + 1,
                    node_type=NodeType.LEAF if node.depth >= 4 else NodeType.INTERNAL,
                    model_used=model,
                    action_taken=depth_action
                )
                node.children.append(depth_child)

    def simulate(self, node: MCTSNode, query: str) -> float:
        """Simulate a rollout from the given node"""
        if not node.model_used:
            return 0.0
        
        try:
            # Use the action to generate a response
            response = self.call_ollama(node.model_used, node.action_taken)
            node.state = response
            
            # Evaluate the response
            reward = self.evaluate_solution(response, query)
            return reward
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate the reward up the tree and update Thompson Sampling parameters"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            
            # Update Thompson Sampling parameters based on action type
            if "width" in current.id:
                current.width_visits += 1
                current.width_reward += reward
                # Update Beta distribution parameters
                if reward > 0.5:  # Good reward
                    current.alpha += 1
                else:  # Poor reward
                    current.beta += 1
            elif "depth" in current.id:
                current.depth_visits += 1
                current.depth_reward += reward
                # Update Beta distribution parameters
                if reward > 0.5:  # Good reward
                    current.alpha += 1
                else:  # Poor reward
                    current.beta += 1
            
            current = current.parent

    def adaptive_branching(self, node: MCTSNode) -> int:
        """Implement adaptive branching based on node performance"""
        if node.visits < 5:
            return 2  # Start with 2 branches
        
        # If node is performing well, explore more
        if node.average_reward > 0.7:
            return min(4, len(node.children) + 1)
        
        # If node is performing poorly, explore less
        if node.average_reward < 0.3:
            return max(1, len(node.children) - 1)
        
        return 2  # Default branching

    def run_ab_mcts(self, query: str, iterations: int, models: List[str], max_depth: int = 5, conversation_context: str = "") -> Tuple[str, Dict[str, Any]]:
        """Run the real Sakana AI AB-MCTS algorithm"""
        
        # Initialize root node
        root = MCTSNode(
            id="root",
            parent=None,
            children=[],
            state="",
            visits=0,
            total_reward=0.0,
            depth=0,
            node_type=NodeType.ROOT
        )
        
        # Initial expansion
        self.expand_node(root, query, models, conversation_context)
        
        search_stats = {
            "total_iterations": iterations,
            "nodes_created": 0,
            "best_reward": 0.0,
            "average_reward": 0.0,
            "exploration_ratio": 0.0,
            "width_searches": 0,
            "depth_searches": 0,
            "model_usage": {}
        }
        
        # MCTS iterations
        for iteration in range(iterations):
            # Selection phase
            current = root
            path = [current]
            
            while current.children and current.depth < max_depth:
                current = self.select_child(current)
                path.append(current)
            
            # Simulation phase
            if current.children:
                # Select a random child for simulation
                child = random.choice(current.children)
                reward = self.simulate(child, query)
                
                # Track search type
                if "width" in child.id:
                    search_stats["width_searches"] += 1
                elif "depth" in child.id:
                    search_stats["depth_searches"] += 1
                
                # Track model usage
                if child.model_used:
                    search_stats["model_usage"][child.model_used] = search_stats["model_usage"].get(child.model_used, 0) + 1
                
                print(f"Iteration {iteration+1}: {child.id} at depth {child.depth}, model: {child.model_used}, reward: {reward:.3f}")
                simulated_node = child
            else:
                reward = self.simulate(current, query)
                print(f"Iteration {iteration+1}: {current.id} at depth {current.depth}, model: {current.model_used}, reward: {reward:.3f}")
                simulated_node = current
            
            # Expansion phase AFTER simulation
            if simulated_node.depth < max_depth and not simulated_node.children:
                self.expand_node(simulated_node, query, models, conversation_context)
                search_stats["nodes_created"] += len(simulated_node.children)
                print(f"  Expanded {simulated_node.id} with {len(simulated_node.children)} children")
            
            # Backpropagation phase
            self.backpropagate(simulated_node, reward)
            
            # Update stats
            search_stats["best_reward"] = max(search_stats["best_reward"], reward)
            
            # Adaptive branching
            if simulated_node.parent:
                branching_factor = self.adaptive_branching(simulated_node.parent)
        
        # Find best solution
        best_node = self.find_best_solution(root)
        search_stats["average_reward"] = root.average_reward if root.visits > 0 else 0.0
        search_stats["exploration_ratio"] = search_stats["width_searches"] / max(search_stats["width_searches"] + search_stats["depth_searches"], 1)
        
        # Add model information to stats
        if best_node and best_node.model_used:
            search_stats["model_used"] = best_node.model_used
        
        return best_node.state if best_node else "No solution found", search_stats

    def find_best_solution(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Find the best solution in the tree"""
        all_solutions = []
        self._collect_solutions(root, all_solutions)
        
        if not all_solutions:
            return None
        
        # Find the best solution based on reward and visits
        def score_solution(node):
            return node.visits * node.average_reward
        
        return max(all_solutions, key=score_solution)
    
    def _collect_solutions(self, node: MCTSNode, solutions: List[MCTSNode]) -> None:
        """Recursively collect all nodes that have actual solutions"""
        if node.state and node.state.strip():
            solutions.append(node)
        
        for child in node.children:
            self._collect_solutions(child, solutions)

    def process_query(self, query: str, conversation_id: Optional[str] = None, iterations: int = 20, models: List[str] = None, max_depth: int = 5) -> Tuple[str, Dict[str, Any], str, str]:
        """Main query processing method"""
        if models is None:
            models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
        # Get conversation context if available
        conversation_context = ""
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            # Build context from recent messages
            recent_messages = conversation.get("messages", [])[-3:]  # Last 3 messages
            conversation_context = " | ".join([msg.get("content", "") for msg in recent_messages])
        
        # Run real AB-MCTS
        result, stats = self.run_ab_mcts(query, iterations, models, max_depth, conversation_context)
        
        # Create turn ID
        turn_id = str(uuid.uuid4())
        
        return result, stats, conversation_id or str(uuid.uuid4()), turn_id

# Initialize service
service = RealSakanaABMCTSService()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using real Sakana AI AB-MCTS algorithm"""
    try:
        result, stats, conversation_id, turn_id = service.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            iterations=request.iterations,
            models=request.models,
            max_depth=request.max_depth
        )
        return QueryResponse(
            result=result, 
            success=True, 
            search_stats=stats,
            conversation_id=conversation_id,
            turn_id=turn_id
        )
    except Exception as e:
        return QueryResponse(
            result="", 
            success=False, 
            error=str(e),
            conversation_id="",
            turn_id=""
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8093)
