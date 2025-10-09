"""
Proper TreeQuest AB-MCTS Service

Uses the actual Sakana AI TreeQuest implementation with proper AB-MCTS algorithm.
Based on their official implementation and TreeQuest documentation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
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
from experiment_logger import ExperimentLogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Proper TreeQuest AB-MCTS Service", version="6.0.0")
# Prometheus metrics
AB_MCTS_QUERIES = Counter(
    "ab_mcts_queries_total",
    "Total AB-MCTS queries",
)
AB_MCTS_SUCCESS = Counter(
    "ab_mcts_success_total",
    "Successful AB-MCTS responses",
)
AB_MCTS_LATENCY = Histogram(
    "ab_mcts_latency_seconds",
    "AB-MCTS end-to-end latency (seconds)",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)
AB_MCTS_ITERATIONS = Histogram(
    "ab_mcts_iterations",
    "Iterations used per query",
    buckets=(1, 3, 5, 10, 15, 20),
)
AB_MCTS_NODES = Histogram(
    "ab_mcts_nodes_created",
    "Nodes created per query",
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000),
)

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
    depth: int = 0  # Track depth in tree (0 = root)

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

        # Load configuration from persistence
        from config_persistence import get_config_persistence
        self.config_persistence = get_config_persistence()
        self._load_config()

        # Initialize experiment logger for research
        self.experiment_logger = ExperimentLogger(base_dir="/app/logs")

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

    def _load_config(self):
        """Load configuration from persistence."""
        config = self.config_persistence.load_config("ab-mcts")

        if config:
            # Load judge models
            self.judge_models = config.get("judge_models", [])
            # Load iterations
            self.default_iterations = config.get("iterations", 20)
            # Load regular models if saved
            saved_models = config.get("models", [])
            if saved_models:
                self.models = saved_models
            # Load criterion weights
            self.criterion_weights = config.get("criterion_weights", {
                "accuracy": 0.25,
                "completeness": 0.25,
                "clarity": 0.25,
                "relevance": 0.25
            })
        else:
            # Defaults
            self.judge_models = []
            self.default_iterations = 20
            self.criterion_weights = {
                "accuracy": 0.25,
                "completeness": 0.25,
                "clarity": 0.25,
                "relevance": 0.25
            }

    def _save_config(self):
        """Save configuration to persistence."""
        config = {
            "models": self.models,
            "judge_models": self.judge_models,
            "iterations": self.default_iterations,
            "criterion_weights": self.criterion_weights
        }
        self.config_persistence.save_config("ab-mcts", config)
        
    def call_ollama(self, model: str, prompt: str, temperature: float = 0.6, max_tokens: int = 1000) -> str:
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
                        "num_predict": max_tokens  # Ollama uses num_predict, not max_tokens
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
    
    def _parse_multi_score(self, response: str) -> Dict[str, float]:
        """Parse multi-criterion scores from judge response.

        Expected format:
        accuracy: 0.85
        completeness: 0.90
        clarity: 0.75
        relevance: 0.80
        """
        import re

        scores = {}
        criteria = ["accuracy", "completeness", "clarity", "relevance"]

        for criterion in criteria:
            # Try to find: "criterion: 0.XX"
            pattern = rf'{criterion}\s*:\s*(\d*\.?\d+)'
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to [0, 1]
                    scores[criterion] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[criterion] = None
            else:
                scores[criterion] = None

        return scores

    def evaluate_solution_quality(self, solution: str, query: str) -> Tuple[float, Dict[str, Any]]:
        """Evaluate solution quality using multi-criterion LLM-as-judge.

        Uses 1 or 2 judge models to evaluate response on 4 criteria:
        - Accuracy: Is it factually correct?
        - Completeness: Does it fully answer the question?
        - Clarity: Is it well-explained and understandable?
        - Relevance: Is it on-topic and directly addresses the query?

        Returns:
            Tuple of (weighted_score, evaluation_details)
        """
        if not solution or "Error:" in solution:
            return 0.0, {"error": "Invalid solution"}

        # If no judge models configured, return neutral score
        if not self.judge_models:
            print("Warning: No judge models configured, using neutral score 0.5")
            return 0.5, {"warning": "No judge models configured"}

        # Multi-criterion prompt with strict formatting
        judge_prompt = f"""Rate this response on each criterion (0.0-1.0):

Query: {query}

Response: {solution}

Evaluate on these criteria:
- Accuracy: Is it factually correct?
- Completeness: Does it fully answer the question?
- Clarity: Is it well-explained and understandable?
- Relevance: Is it on-topic and addresses the query?

Output EXACTLY in this format (replace X.XX with your scores):
accuracy: X.XX
completeness: X.XX
clarity: X.XX
relevance: X.XX

Example:
accuracy: 0.85
completeness: 0.90
clarity: 0.75
relevance: 0.95

Your ratings:
"""

        # Collect scores from each judge
        all_criterion_scores = {
            "accuracy": [],
            "completeness": [],
            "clarity": [],
            "relevance": []
        }
        judge_details = {}

        # Query each judge model
        for judge_model in self.judge_models:
            try:
                print(f"[JUDGE] Calling {judge_model} for multi-criterion evaluation")

                # Token allocation for multi-criterion (need space for 4 scores)
                max_tokens = min(300 + (len(solution) // 20), 500)

                response = self.call_ollama(
                    judge_model,
                    judge_prompt,
                    temperature=0.1,
                    max_tokens=max_tokens
                )

                print(f"[JUDGE] Response ({len(response)} chars): {response[:200]}")

                # Parse multi-criterion scores
                criterion_scores = self._parse_multi_score(response)
                print(f"[JUDGE] Parsed scores: {criterion_scores}")

                # Validate that we got scores
                valid_scores = {k: v for k, v in criterion_scores.items() if v is not None}

                if len(valid_scores) >= 2:  # Need at least 2 criteria scored
                    # Collect scores for averaging across judges
                    for criterion, score in valid_scores.items():
                        all_criterion_scores[criterion].append(score)

                    judge_details[judge_model] = {
                        "breakdown": criterion_scores,
                        "raw_response": response.strip(),
                        "num_criteria_scored": len(valid_scores)
                    }
                    print(f"[JUDGE] Successfully scored {len(valid_scores)} criteria")
                else:
                    # Fallback: try to extract a single overall score
                    print(f"[JUDGE] Multi-score parsing failed, trying single score extraction")
                    single_score = self._extract_score(response)
                    if single_score is not None:
                        # Use same score for all criteria
                        for criterion in all_criterion_scores.keys():
                            all_criterion_scores[criterion].append(single_score)
                        judge_details[judge_model] = {
                            "breakdown": {k: single_score for k in all_criterion_scores.keys()},
                            "raw_response": response.strip(),
                            "fallback_mode": True
                        }
                    else:
                        judge_details[judge_model] = {
                            "error": "Could not parse scores",
                            "raw_response": response.strip()
                        }

            except Exception as e:
                print(f"[JUDGE] Exception calling {judge_model}: {type(e).__name__}: {e}")
                judge_details[judge_model] = {
                    "error": str(e)
                }

        # Average scores across judges for each criterion
        criterion_averages = {}
        for criterion, scores_list in all_criterion_scores.items():
            if scores_list:
                criterion_averages[criterion] = sum(scores_list) / len(scores_list)
            else:
                criterion_averages[criterion] = None

        # Calculate weighted final score
        valid_criteria = {k: v for k, v in criterion_averages.items() if v is not None}

        if valid_criteria:
            # Apply weights (normalize if some criteria missing)
            total_weight = sum(self.criterion_weights.get(k, 0.25) for k in valid_criteria.keys())

            weighted_score = sum(
                score * (self.criterion_weights.get(criterion, 0.25) / total_weight)
                for criterion, score in valid_criteria.items()
            )

            return weighted_score, {
                "criterion_scores": criterion_averages,
                "weights": self.criterion_weights,
                "final_score": weighted_score,
                "num_judges": len([j for j in judge_details.values() if "error" not in j]),
                "judge_details": judge_details
            }

        # Fallback if all judges failed
        print("[JUDGE] All judges failed, using neutral score 0.5")
        return 0.5, {
            "error": "All judges failed",
            "fallback_score": 0.5,
            "judge_details": judge_details
        }

    def _extract_score(self, response: str) -> Optional[float]:
        """Extract numeric score from judge response.

        Args:
            response: Judge model response

        Returns:
            Score between 0.0 and 1.0, or None if not found
        """
        import re

        # Remove whitespace
        response = response.strip()

        # Try to find a number (decimal or whole)
        # Priority order: most specific to most general
        patterns = [
            r'<score>(\d*\.?\d+)</score>',  # XML tag format (our preferred format)
            r'<score>(\d*\.?\d+)',  # Incomplete closing tag
            r'score[:\s]+(\d*\.?\d+)',  # Score: 0.85
            r'</think>\s*(\d*\.?\d+)',  # After closing think tag
            r'</think>\s*score[:\s]*(\d*\.?\d+)',  # After think, then Score:
            r'^(\d*\.?\d+)$',  # Just a number: 0.85
            r'^(\d+)%$',  # Percentage: 85%
            r'(\d*\.?\d+)\s*out of\s*1',  # X out of 1
            r'(\d*\.?\d+)\s*/\s*1',  # X/1
            r'rating[:\s]+(\d*\.?\d+)',  # Rating: 0.85
            r'(\d*\.?\d+)\s*/\s*1\.?0',  # X/1.0
            # Fallback: find ANY decimal number between 0 and 1
            r'\b(0\.\d+|1\.0+|1)\b',  # Any decimal 0.x or 1.0
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE | re.DOTALL)
            if match:
                try:
                    score = float(match.group(1))

                    # Convert percentage to decimal
                    if '%' in response:
                        score = score / 100

                    # Clamp to [0, 1]
                    score = max(0.0, min(1.0, score))

                    # Sanity check: scores outside [0,1] before clamping indicate bad parse
                    if score > 1.5:  # Likely parsed wrong number
                        continue

                    return score

                except (ValueError, IndexError):
                    continue

        return None

    def _extract_tree_structure(self, search_tree) -> Dict[str, Any]:
        """Extract hierarchical tree structure with parent-child relationships.

        Args:
            search_tree: The TreeQuest search tree (ABMCTSMState)

        Returns:
            Dictionary representing the tree hierarchy
        """
        try:
            # Access the actual tree from ABMCTSMState
            tree = search_tree.tree if hasattr(search_tree, 'tree') else search_tree
            root_node = tree.root if hasattr(tree, 'root') else None

            if not root_node:
                return {"type": "root", "children": []}

            # Recursively build tree structure
            node_id_counter = [0]  # Use list to make it mutable in nested function

            def build_node_dict(node, depth=0) -> Dict[str, Any]:
                node_id = node_id_counter[0]
                node_id_counter[0] += 1

                node_dict = {
                    "id": node_id,
                    "depth": depth,
                    "score": getattr(node, 'score', 0.0),
                    "children": []
                }

                # Add state info if available
                if hasattr(node, 'state') and node.state is not None:
                    state = node.state
                    node_dict.update({
                        "model": getattr(state, 'model_name', 'unknown'),
                        "search_type": getattr(state, 'search_type', 'unknown'),
                        "quality": getattr(state, 'eval_results', {}).get('quality', 0.0),
                        "preview": (getattr(state, 'generation_result', '') or '')[:80]
                    })

                # Recursively add children
                if hasattr(node, 'children'):
                    for child in node.children:
                        node_dict["children"].append(build_node_dict(child, depth + 1))

                return node_dict

            return build_node_dict(root_node)

        except Exception as e:
            print(f"Error extracting tree structure: {e}")
            return {"type": "root", "children": [], "error": str(e)}

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

        # Determine current depth and search type
        current_depth = state.depth if state is not None else 0

        if state is None or not state.generation_result:
            # Width search - generate new solution
            prompt = self.generate_width_prompt(query)
            search_type = "width"
            new_depth = 1
        else:
            # Depth search - refine existing solution
            prompt = self.generate_depth_prompt(state.generation_result, query)
            search_type = "depth"
            new_depth = current_depth + 1

        # Get response from model
        response = self.call_ollama(model_name, prompt, temperature=0.6)

        # Evaluate quality with judge details
        quality, judge_details = self.evaluate_solution_quality(response, query)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create node state with full details
        node_state = NodeState(
            generation_result=response,
            eval_results={
                "quality": quality,
                "search_type": search_type,
                "judge_details": judge_details,
                "execution_time": execution_time,
                "timestamp": time.time()
            },
            model_name=model_name,
            search_type=search_type,
            depth=new_depth
        )

        return node_state, quality
    
    def run_proper_treequest_ab_mcts(self, query: str, iterations: int = 20, include_tree: bool = False) -> Dict[str, Any]:
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

        for i in range(iterations):  # Use full iterations value
            # Run one step of AB-MCTS
            search_tree = self.algo.step(search_tree, generate_fns)

            # Get current state-score pairs
            state_score_pairs = self.algo.get_state_score_pairs(search_tree)
            search_stats["nodes_created"] = len(state_score_pairs)

            # Extract tree structure with parent-child relationships
            tree_structure = self._extract_tree_structure(search_tree)

            # ALWAYS capture full tree data for research (not just when include_tree=True)
            # Store complete node information with full responses
            iteration_log.append({
                "iteration": i + 1,
                "timestamp": time.time(),
                "nodes": [
                    {
                        "model": getattr(state, 'model_name', getattr(state, 'model_used', 'unknown')),
                        "quality": score,
                        "search_type": getattr(state, 'search_type', getattr(state, 'eval_results', {}).get('search_type', 'unknown')),
                        "full_response": getattr(state, 'generation_result', getattr(state, 'content', '')),  # FULL response, not preview
                        "eval_results": getattr(state, 'eval_results', {}),  # Includes judge details
                        "preview": (getattr(state, 'generation_result', getattr(state, 'content', '')) or '')[:120]  # Keep preview for UI
                    }
                    for state, score in state_score_pairs[:100]  # Limit to 100 nodes per iteration for performance
                ],
                "tree_structure": tree_structure  # Add actual tree hierarchy
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

        # Return full iteration log for experiment logging (UI can filter if needed)
        return {
            "solution": best_solution,
            "search_stats": search_stats,
            "iteration_log": iteration_log,  # Always include for research
            "final_best_score": best_score
        }
    
    def update_models(self, model_names: List[str]) -> bool:
        """Update the models used for AB-MCTS."""
        try:
            # Validate models are available
            validation = self.model_discovery.validate_models(model_names)
            unavailable = [name for name, available in validation.items() if not available]

            if unavailable:
                print(f"Warning: Some models not available: {unavailable}")

            # Update models list
            self.models = [name for name in model_names if validation.get(name, False)]

            if not self.models:
                print("Error: No valid models selected")
                return False

            # Persist configuration
            self._save_config()

            print(f"Updated models: {self.models}")
            return True

        except Exception as e:
            print(f"Error updating models: {e}")
            return False

    def update_judge_models(self, judge_model_names: List[str]) -> bool:
        """Update the judge models used for evaluation.

        Args:
            judge_model_names: List of 1-2 judge model names

        Returns:
            True if successful, False otherwise
        """
        try:
            # Limit to 2 judges
            if len(judge_model_names) > 2:
                print("Warning: Maximum 2 judge models allowed, taking first 2")
                judge_model_names = judge_model_names[:2]

            # Validate models are available
            validation = self.model_discovery.validate_models(judge_model_names)
            unavailable = [name for name, available in validation.items() if not available]

            if unavailable:
                print(f"Warning: Some judge models not available: {unavailable}")

            # Update judge models list
            self.judge_models = [name for name in judge_model_names if validation.get(name, False)]

            # Persist configuration
            self._save_config()

            print(f"Updated judge models: {self.judge_models}")
            return True

        except Exception as e:
            print(f"Error updating judge models: {e}")
            return False

    def update_search_params(self, iterations: Optional[int] = None) -> bool:
        """Update default search parameters.

        Args:
            iterations: Default iterations (1-100). More iterations allow Thompson sampling
                       to naturally explore deeper trees through exploitation.

        Returns:
            True if successful, False otherwise
        """
        try:
            if iterations is not None:
                self.default_iterations = max(1, min(100, iterations))

            # Persist configuration
            self._save_config()

            print(f"Updated search params: iterations={self.default_iterations}")
            return True

        except Exception as e:
            print(f"Error updating search params: {e}")
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
    
    def process_query(self, query: str, iterations: int = 20,
                     conversation_id: Optional[str] = None, models: Optional[List[str]] = None) -> QueryResponse:
        """Process a query using proper TreeQuest AB-MCTS."""
        run_id = None
        try:
            AB_MCTS_QUERIES.inc()

            # Start experiment logging
            run_id = self.experiment_logger.start_run(
                pipeline="ab-mcts",
                user_query=query,
                parameters={
                    "iterations": iterations,
                    "models": self.models,
                    "judge_models": self.judge_models
                },
                metadata={
                    "conversation_id": conversation_id or str(uuid.uuid4())
                }
            )

            # Update models if provided
            if models:
                self.update_models(models)

            # Run proper TreeQuest AB-MCTS search
            include_tree = True  # always capture for research; UI can ignore
            t0 = time.time()
            result = self.run_proper_treequest_ab_mcts(query, iterations, include_tree)
            AB_MCTS_LATENCY.observe(time.time() - t0)

            # Log each iteration to the experiment log
            for iteration_data in result.get("iteration_log", []):
                self.experiment_logger.log_event(run_id, {
                    "type": "iteration",
                    "iteration": iteration_data["iteration"],
                    "timestamp": iteration_data.get("timestamp"),
                    "node_count": len(iteration_data["nodes"]),
                    "nodes": iteration_data["nodes"],  # Full nodes with complete responses
                    "tree_structure": iteration_data.get("tree_structure", {})  # Include hierarchical tree structure
                })

            # Log final result
            self.experiment_logger.finish_run(run_id, {
                "solution": result["solution"],
                "search_stats": result["search_stats"],
                "final_best_score": result.get("final_best_score", 0.0),
                "total_nodes_explored": len(result.get("iteration_log", []))
            })

            # Create response
            response = QueryResponse(
                result=result["solution"],
                success=True,
                search_stats=result["search_stats"],
                conversation_id=conversation_id or str(uuid.uuid4()),
                turn_id=str(uuid.uuid4()),
                iteration_log=result.get("iteration_log", [])
            )

            # Metrics from search stats
            stats = result["search_stats"]
            AB_MCTS_ITERATIONS.observe(stats.get("total_iterations", 0) or 0)
            AB_MCTS_NODES.observe(stats.get("nodes_created", 0) or 0)
            AB_MCTS_SUCCESS.inc()
            return response

        except Exception as e:
            # Log failure to experiment log
            if run_id:
                self.experiment_logger.fail_run(run_id, str(e))

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

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/query")
async def process_query_endpoint(request: QueryRequest):
    """Process a query using proper TreeQuest AB-MCTS."""
    try:
        # Use saved defaults if not explicitly provided in request
        iterations = request.iterations if request.iterations is not None else service.default_iterations

        response = service.process_query(
            query=request.query,
            iterations=iterations,
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
            "current_models": service.models,
            "judge_models": service.judge_models
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
            iterations=5
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

@app.post("/judges/update")
async def update_judges(request: Dict[str, Any]):
    """Update the judge models used for evaluation."""
    try:
        judge_model_names = request.get("judge_models", [])
        success = service.update_judge_models(judge_model_names)

        return {
            "success": success,
            "message": "Judge models updated successfully" if success else "Failed to update judge models",
            "judge_models": service.judge_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/params/update")
async def update_params(request: Dict[str, Any]):
    """Update default search parameters."""
    try:
        iterations = request.get("iterations")

        success = service.update_search_params(iterations=iterations)

        return {
            "success": success,
            "message": "Search parameters updated successfully. Tree depth controlled by Thompson sampling." if success else "Failed to update search parameters",
            "iterations": service.default_iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/weights/update")
async def update_weights(request: Dict[str, Any]):
    """Update criterion weights for judge evaluation."""
    try:
        weights = request.get("weights", {})

        # Validate weights
        required_criteria = ["accuracy", "completeness", "clarity", "relevance"]
        for criterion in required_criteria:
            if criterion not in weights:
                raise HTTPException(status_code=400, detail=f"Missing weight for '{criterion}'")

            weight = weights[criterion]
            if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                raise HTTPException(status_code=400, detail=f"Invalid weight for '{criterion}': must be 0.0-1.0")

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total == 0:
            raise HTTPException(status_code=400, detail="Weights cannot all be zero")

        normalized_weights = {k: v / total for k, v in weights.items()}

        # Update and save
        service.criterion_weights = normalized_weights
        service._save_config()

        return {
            "success": True,
            "message": "Criterion weights updated successfully",
            "weights": service.criterion_weights
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "success": True,
        "models": service.models,
        "judge_models": service.judge_models,
        "iterations": service.default_iterations,
        "criterion_weights": service.criterion_weights
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "service": "proper-treequest-ab-mcts",
        "status": "running",
        "models": service.models,
        "judge_models": service.judge_models,
        "default_iterations": service.default_iterations,
        "algo_config": service.algo_config
    }

@app.get("/runs")
async def list_runs(limit: int = 50):
    """List recent experiment runs."""
    try:
        runs = service.experiment_logger.list_runs(limit=limit)
        return {
            "success": True,
            "runs": runs,
            "count": len(runs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get detailed information about a specific run."""
    try:
        run = service.experiment_logger.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get event stream (full tree data)
        events = service.experiment_logger.read_events(run_id)

        return {
            "success": True,
            "run": run,
            "events": events,
            "event_count": len(events)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}/tree")
async def get_run_tree(run_id: str):
    """Get the tree visualization data for a specific run."""
    try:
        run = service.experiment_logger.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get all iteration events
        events = service.experiment_logger.read_events(run_id)
        print(f"[DEBUG /tree] Total events: {len(events)}")

        # Check if this is old format (iterations array) or new format (tree_structure)
        iteration_events = [e for e in events if e.get("type") == "iteration"]
        print(f"[DEBUG /tree] Iteration events: {len(iteration_events)}")

        if not iteration_events:
            raise HTTPException(status_code=404, detail="No iteration data found for this run")

        # Check if we have tree_structure in the last iteration
        last_iteration = iteration_events[-1] if iteration_events else {}
        print(f"[DEBUG /tree] Last iteration keys: {list(last_iteration.keys())}")
        has_tree_structure = "tree_structure" in last_iteration
        print(f"[DEBUG /tree] Has tree_structure: {has_tree_structure}")

        if has_tree_structure:
            # NEW FORMAT: Use tree_structure from last iteration
            final_tree_structure = last_iteration.get("tree_structure", {"type": "root", "children": []})
            all_nodes = last_iteration.get("nodes", [])
            total_iterations = last_iteration.get("iteration", len(iteration_events))

            tree_data = {
                "run_id": run_id,
                "query": run.get("user_query"),
                "parameters": run.get("parameters"),
                "total_iterations": total_iterations,
                "total_nodes": len(all_nodes),
                "tree_structure": final_tree_structure,
                "nodes": all_nodes
            }
        else:
            # OLD FORMAT: Return iterations array for backward compatibility
            tree_data = {
                "run_id": run_id,
                "query": run.get("user_query"),
                "parameters": run.get("parameters"),
                "iterations": [
                    {
                        "iteration": e.get("iteration"),
                        "timestamp": e.get("timestamp"),
                        "node_count": e.get("node_count"),
                        "nodes": e.get("nodes", []),
                        "tree_structure": {"type": "root", "children": []}  # Empty for old format
                    }
                    for e in iteration_events
                ]
            }

        return {
            "success": True,
            "tree_data": tree_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)
