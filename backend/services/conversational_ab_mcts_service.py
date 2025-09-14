"""
Conversational AB-MCTS Multi-Model Service

This implements AB-MCTS with conversation threading, allowing follow-up questions
and iterative problem-solving through multiple turns.

Based on Sakana AI's research on Adaptive Branching Monte Carlo Tree Search.
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
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

app = FastAPI(title="Conversational AB-MCTS Multi-Model Service", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NodeType(Enum):
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"

@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree"""
    id: str
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    state: str  # The current text/solution state
    visits: int
    total_reward: float
    depth: int
    node_type: NodeType
    model_used: Optional[str] = None
    action_taken: Optional[str] = None
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)
    
    @property
    def ucb_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.average_reward
        
        # UCB1 formula with exploration constant
        exploration_constant = 1.414  # sqrt(2)
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    user_message: str
    ab_mcts_result: str
    timestamp: datetime
    search_stats: Dict[str, Any]

@dataclass
class Conversation:
    """Represents a conversation thread"""
    conversation_id: str
    turns: List[ConversationTurn]
    context_summary: str
    created_at: datetime
    last_updated: datetime

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None  # If provided, continues existing conversation
    iterations: Optional[int] = 20
    models: Optional[List[str]] = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
    max_depth: Optional[int] = 5
    exploration_constant: Optional[float] = 1.414

class QueryResponse(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None
    search_stats: Optional[Dict[str, Any]] = None
    conversation_id: str
    turn_id: str

class ConversationalABMCTSService:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.conversations: Dict[str, Conversation] = {}
        
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

    def evaluate_solution(self, solution: str, original_query: str) -> float:
        """Evaluate the quality of a solution"""
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

    def generate_actions(self, current_state: str, query: str, model: str, conversation_context: str = "") -> List[str]:
        """Generate possible actions/improvements to the current state"""
        actions = []
        
        # Build context-aware prompt
        context_prompt = ""
        if conversation_context:
            context_prompt = f"Previous conversation context: {conversation_context}\n\n"
        
        if not current_state or current_state.strip() == "":
            # No previous state - generate initial responses
            actions.append(f"{context_prompt}Answer this question directly and comprehensively: {query}")
            actions.append(f"{context_prompt}Provide a detailed explanation of: {query}")
            actions.append(f"{context_prompt}Give a thorough analysis of: {query}")
            actions.append(f"{context_prompt}Explain {query} with examples and context")
            actions.append(f"{context_prompt}Create a comprehensive response about: {query}")
            print(f"  Generating initial actions for {model} (no previous state)")
        else:
            # Has previous state - generate improvements
            actions.append(f"{context_prompt}Expand and elaborate on this response: {current_state}")
            actions.append(f"{context_prompt}Add specific examples and details to this response: {current_state}")
            actions.append(f"{context_prompt}Reorganize and improve the structure of this response: {current_state}")
            print(f"  Generating improvement actions for {model} (building on previous state)")
            actions.append(f"{context_prompt}Address different aspects and perspectives of: {query}")
            actions.append(f"{context_prompt}Synthesize this with broader knowledge about: {query}")
        
        return actions

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB1"""
        if not node.children:
            return node
        
        # Select child with highest UCB value
        best_child = max(node.children, key=lambda x: x.ucb_value)
        return best_child

    def expand_node(self, node: MCTSNode, query: str, models: List[str], conversation_context: str = "") -> None:
        """Expand a node by adding children"""
        if node.depth >= 5:  # Max depth limit
            return
        
        # Generate actions for each model
        for model in models:
            actions = self.generate_actions(node.state, query, model, conversation_context)
            
            for action in actions[:2]:  # Limit to 2 actions per model
                # Create child node
                child_id = f"{node.id}_{len(node.children)}_{model}"
                child = MCTSNode(
                    id=child_id,
                    parent=node,
                    children=[],
                    state="",  # Will be filled during simulation
                    visits=0,
                    total_reward=0.0,
                    depth=node.depth + 1,
                    node_type=NodeType.LEAF if node.depth >= 4 else NodeType.INTERNAL,
                    model_used=model,
                    action_taken=action
                )
                node.children.append(child)

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
        """Backpropagate the reward up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
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
        """Run the actual AB-MCTS algorithm"""
        
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
            "exploration_ratio": 0.0
        }
        
        # MCTS iterations
        for iteration in range(iterations):
            # Selection phase
            current = root
            path = [current]
            
            while current.children and current.depth < max_depth:
                current = self.select_child(current)
                path.append(current)
            
            # Simulation phase FIRST - get a response
            if current.children:
                # Select a random child for simulation
                child = random.choice(current.children)
                reward = self.simulate(child, query)
                print(f"Iteration {iteration+1}: Simulating child at depth {child.depth}, model: {child.model_used}, reward: {reward:.3f}")
                simulated_node = child
            else:
                reward = self.simulate(current, query)
                print(f"Iteration {iteration+1}: Simulating leaf at depth {current.depth}, model: {current.model_used}, reward: {reward:.3f}")
                simulated_node = current
            
            # Expansion phase AFTER simulation - now we have a response to build on
            if simulated_node.depth < max_depth and not simulated_node.children:
                self.expand_node(simulated_node, query, models, conversation_context)
                search_stats["nodes_created"] += len(simulated_node.children)
                print(f"  Expanded node at depth {simulated_node.depth} with {len(simulated_node.children)} children")
            
            # Backpropagation phase
            self.backpropagate(simulated_node, reward)
            
            # Update stats
            search_stats["best_reward"] = max(search_stats["best_reward"], reward)
            
            # Adaptive branching
            if simulated_node.parent:
                branching_factor = self.adaptive_branching(simulated_node.parent)
                # This would be used in future expansions
        
        # Find best solution
        best_node = self.find_best_solution(root)
        search_stats["average_reward"] = root.average_reward if root.visits > 0 else 0.0
        
        # Add model information to stats
        if best_node and best_node.model_used:
            search_stats["model_used"] = best_node.model_used
        
        return best_node.state if best_node else "No solution found", search_stats

    def find_best_solution(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Find the best solution in the tree"""
        # Collect all nodes with actual solutions (state is not empty)
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
        if node.state and node.state.strip():  # Node has a solution
            solutions.append(node)
        
        for child in node.children:
            self._collect_solutions(child, solutions)

    def create_conversation_context(self, conversation: Conversation) -> str:
        """Create a context summary from the conversation history"""
        if not conversation.turns:
            return ""
        
        # Get the last few turns for context
        recent_turns = conversation.turns[-3:]  # Last 3 turns
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Q: {turn.user_message}")
            context_parts.append(f"A: {turn.ab_mcts_result[:200]}...")  # Truncate long responses
        
        return "\n".join(context_parts)

    def process_query(self, query: str, conversation_id: Optional[str] = None, iterations: int = 20, models: List[str] = None, max_depth: int = 5) -> Tuple[str, Dict[str, Any], str, str]:
        """Main query processing method with conversation support"""
        if models is None:
            models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            conversation_context = self.create_conversation_context(conversation)
        else:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                conversation_id=conversation_id,
                turns=[],
                context_summary="",
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            self.conversations[conversation_id] = conversation
            conversation_context = ""
        
        # Run AB-MCTS with conversation context
        result, stats = self.run_ab_mcts(query, iterations, models, max_depth, conversation_context)
        
        # Create turn ID
        turn_id = str(uuid.uuid4())
        
        # Add turn to conversation
        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=query,
            ab_mcts_result=result,
            timestamp=datetime.now(),
            search_stats=stats
        )
        conversation.turns.append(turn)
        conversation.last_updated = datetime.now()
        conversation.context_summary = self.create_conversation_context(conversation)
        
        return result, stats, conversation_id, turn_id

# Initialize service
service = ConversationalABMCTSService()

@app.get("/")
async def root():
    return {"message": "Conversational AB-MCTS Multi-Model Service", "status": "running", "version": "3.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/conversations")
async def list_conversations():
    """List all conversations"""
    return {
        "conversations": [
            {
                "conversation_id": conv_id,
                "turns_count": len(conv.turns),
                "created_at": conv.created_at.isoformat(),
                "last_updated": conv.last_updated.isoformat()
            }
            for conv_id, conv in service.conversations.items()
        ]
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    if conversation_id not in service.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = service.conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "user_message": turn.user_message,
                "ab_mcts_result": turn.ab_mcts_result,
                "timestamp": turn.timestamp.isoformat(),
                "search_stats": turn.search_stats
            }
            for turn in conversation.turns
        ],
        "created_at": conversation.created_at.isoformat(),
        "last_updated": conversation.last_updated.isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using conversational AB-MCTS algorithm"""
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
    uvicorn.run(app, host="0.0.0.0", port=8092)
