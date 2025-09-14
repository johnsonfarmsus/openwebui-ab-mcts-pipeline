"""
AB-MCTS Multi-Model Function for Open WebUI

This function implements Adaptive Branching Monte Carlo Tree Search (AB-MCTS)
using multiple small language models to collaboratively solve problems.

Requirements: requests, treequest[abmcts-m]
"""

import requests
import json
import subprocess
import sys
import os
from typing import List, Dict, Any, Generator
from pydantic import BaseModel, Field

class ABMCTSFunction:
    def __init__(self):
        self.name = "ab_mcts_multi_model"
        self.description = "Run AB-MCTS with multiple models (deepseek-r1:1.5b, gemma3:1b, llama3.2:1b) for collaborative problem solving"
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or problem to solve using AB-MCTS"
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of MCTS iterations (default: 10)",
                    "default": 10
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of models to use (default: deepseek-r1:1.5b, gemma3:1b, llama3.2:1b)",
                    "default": ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
                },
                "enable_ab_mcts": {
                    "type": "boolean",
                    "description": "Enable full AB-MCTS algorithm (default: true)",
                    "default": True
                }
            },
            "required": ["query"]
        }
        self.ollama_base_url = "http://host.docker.internal:11434"
        self._install_dependencies()

    def _install_dependencies(self):
        """Install required dependencies"""
        try:
            import treequest
            print("TreeQuest already available")
        except ImportError:
            print("Installing TreeQuest...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "treequest[abmcts-m]", "--no-cache-dir"
                ])
                print("TreeQuest installed successfully")
            except Exception as e:
                print(f"Failed to install TreeQuest: {e}")

    def call(self, query: str, iterations: int = 10, models: List[str] = None, enable_ab_mcts: bool = True) -> str:
        """Execute AB-MCTS with multiple models"""
        if models is None:
            models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        
        if enable_ab_mcts:
            try:
                return self._run_ab_mcts(query, iterations, models)
            except Exception as e:
                return f"AB-MCTS failed ({str(e)}), falling back to simple multi-model approach...\n\n" + self._simple_multi_model(query, models)
        else:
            return self._simple_multi_model(query, models)

    def _run_ab_mcts(self, query: str, iterations: int, models: List[str]) -> str:
        """Run full AB-MCTS implementation"""
        try:
            import treequest as tq
            
            result = []
            result.append("🌳 **AB-MCTS Multi-Model Search**\n")
            result.append(f"**Query:** {query}\n")
            result.append(f"**Models:** {', '.join(models)}\n")
            result.append(f"**Iterations:** {iterations}\n\n")
            result.append("---\n\n")

            # Create model generators
            model_generators = {}
            for model_name in models:
                model_generators[f"{model_name}_gen"] = self._create_generator(model_name, query)

            # Initialize AB-MCTS
            algo = tq.ABMCTSA()
            search_tree = algo.init_tree()

            result.append("🚀 **Starting search process...**\n\n")

            # Run the search
            for i in range(iterations):
                search_tree = algo.step(search_tree, model_generators)
                
                if (i + 1) % 3 == 0:  # Progress every 3 iterations
                    try:
                        best_state, best_score = tq.top_k(search_tree, algo, k=1)[0]
                        result.append(f"**Iteration {i+1}/{iterations}** - Best Score: {best_score:.3f}\n")
                        result.append(f"*Preview:* {best_state.content[:150]}{'...' if len(best_state.content) > 150 else ''}\n\n")
                    except Exception as e:
                        result.append(f"*Iteration {i+1}: Processing...*\n\n")

            # Get final results
            try:
                top_results = tq.top_k(search_tree, algo, k=3)
                
                result.append("## 🏆 **Final Results**\n\n")
                
                for rank, (state, score) in enumerate(top_results, 1):
                    result.append(f"### Rank {rank} (Score: {score:.3f})\n")
                    result.append(f"**Model:** {getattr(state, 'model_used', 'Unknown')}\n\n")
                    result.append(f"{state.content}\n\n")
                    result.append("---\n\n")
                    
            except Exception as e:
                result.append(f"Error getting results: {str(e)}\n")
                
            return "".join(result)
                
        except ImportError:
            raise Exception("TreeQuest not available")
        except Exception as e:
            raise Exception(f"AB-MCTS execution failed: {str(e)}")

    def _simple_multi_model(self, query: str, models: List[str]) -> str:
        """Fallback multi-model approach without TreeQuest"""
        
        result = []
        result.append("🤖 **Multi-Model Collaboration** (Simplified)\n\n")
        result.append(f"**Query:** {query}\n\n")

        responses = []
        
        # Get responses from all models
        for i, model in enumerate(models):
            result.append(f"### Consulting {model}...\n")
            
            try:
                response = self._call_ollama(model, query)
                responses.append({"model": model, "response": response})
                result.append(f"**Response:** {response[:200]}{'...' if len(response) > 200 else ''}\n\n")
            except Exception as e:
                result.append(f"**Error:** {str(e)}\n\n")

        # Collaborative refinement
        if len(responses) > 1:
            result.append("---\n\n## 🔄 **Collaborative Refinement**\n\n")
            
            for model_data in responses:
                other_responses = [r["response"] for r in responses if r["model"] != model_data["model"]]
                context = "\n".join([f"- {resp[:100]}..." for resp in other_responses])
                
                refinement_prompt = f"""Original question: {query}

Other models suggested:
{context}

Considering these perspectives, provide your refined response:"""

                try:
                    refined = self._call_ollama(model_data["model"], refinement_prompt)
                    result.append(f"### {model_data['model']} (Refined)\n")
                    result.append(f"{refined}\n\n")
                except Exception as e:
                    result.append(f"### {model_data['model']} (Refinement failed)\n")
                    result.append(f"Error: {str(e)}\n\n")

        return "".join(result)

    def _create_generator(self, model_name: str, base_prompt: str):
        """Create a generator function for AB-MCTS"""
        def generate(parent_state=None):
            try:
                if parent_state is None:
                    prompt = base_prompt
                else:
                    prompt = f"Previous attempt: {parent_state.content}\n\nImprove this response to: {base_prompt}"
                
                response = self._call_ollama(model_name, prompt)
                score = self._calculate_score(response)
                
                # Create a simple state object
                state = type('State', (), {
                    'content': response,
                    'model_used': model_name
                })()
                
                return state, score
                
            except Exception as e:
                error_state = type('State', (), {
                    'content': f"Error: {str(e)}",
                    'model_used': model_name
                })()
                return error_state, 0.0
        
        return generate

    def _call_ollama(self, model: str, prompt: str) -> str:
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

    def _calculate_score(self, content: str) -> float:
        """Simple scoring mechanism"""
        if not content or "Error:" in content:
            return 0.0
            
        # Basic scoring based on length and content quality
        base_score = min(len(content) / 500, 1.0)
        
        # Bonus for structured responses
        if any(marker in content for marker in ["1.", "2.", "First", "Second", "Therefore"]):
            base_score *= 1.2
            
        # Penalty for uncertainty
        if any(word in content.lower() for word in ["sorry", "don't know", "unclear"]):
            base_score *= 0.7
            
        return min(base_score, 1.0)

# Create the function instance
ab_mcts_function = ABMCTSFunction()