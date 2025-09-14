"""
title: AB-MCTS Multi-Model Pipeline
author: open-webui
date: 2025-09-13
version: 1.0
license: MIT
description: A pipeline that implements AB-MCTS with multiple small models
requirements: requests, treequest[abmcts-m]
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import json
import subprocess
import sys
import os

class Pipeline:
    class Valves(BaseModel):
        model_list: List[str] = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
        iterations: int = 10
        ollama_base_url: str = "http://host.docker.internal:11434"
        enable_ab_mcts: bool = True

    def __init__(self):
        self.type = "pipe"
        self.id = "ab_mcts_pipeline"
        self.name = "AB-MCTS Multi-Model"
        self.valves = self.Valves()
        self._install_dependencies()

    def _install_dependencies(self):
        """Install TreeQuest and other dependencies"""
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

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline execution"""
        
        if not self.valves.enable_ab_mcts:
            # Fallback to simple multi-model approach
            yield from self._simple_multi_model(user_message)
            return

        try:
            # Try AB-MCTS with TreeQuest
            yield from self._run_ab_mcts(user_message)
        except Exception as e:
            yield f"⚠️ AB-MCTS failed ({str(e)}), falling back to simple multi-model...\n\n"
            yield from self._simple_multi_model(user_message)

    def _run_ab_mcts(self, user_message: str) -> Generator[str, None, None]:
        """Run full AB-MCTS implementation"""
        try:
            import treequest as tq
            
            yield "🌳 **AB-MCTS Multi-Model Search**\n\n"
            yield f"**Query:** {user_message}\n\n"
            yield f"**Models:** {', '.join(self.valves.model_list)}\n"
            yield f"**Iterations:** {self.valves.iterations}\n\n"
            yield "---\n\n"

            # Create model generators
            models = {}
            for model_name in self.valves.model_list:
                models[f"{model_name}_gen"] = self._create_generator(model_name, user_message)

            # Initialize AB-MCTS
            algo = tq.ABMCTSA()  # or ABMCTSM for mixed approach
            search_tree = algo.init_tree()

            yield "🚀 **Starting search process...**\n\n"

            # Run the search
            for i in range(self.valves.iterations):
                search_tree = algo.step(search_tree, models)
                
                if (i + 1) % 3 == 0:  # Progress every 3 iterations
                    try:
                        best_state, best_score = tq.top_k(search_tree, algo, k=1)[0]
                        yield f"**Iteration {i+1}/{self.valves.iterations}** - Best Score: {best_score:.3f}\n"
                        yield f"*Preview:* {best_state.content[:150]}{'...' if len(best_state.content) > 150 else ''}\n\n"
                    except Exception as e:
                        yield f"*Iteration {i+1}: Processing...*\n\n"

            # Get final results
            try:
                top_results = tq.top_k(search_tree, algo, k=3)
                
                yield "## 🏆 **Final Results**\n\n"
                
                for rank, (state, score) in enumerate(top_results, 1):
                    yield f"### Rank {rank} (Score: {score:.3f})\n"
                    yield f"**Model:** {getattr(state, 'model_used', 'Unknown')}\n\n"
                    yield f"{state.content}\n\n"
                    yield "---\n\n"
                    
            except Exception as e:
                yield f"Error getting results: {str(e)}\n"
                
        except ImportError:
            raise Exception("TreeQuest not available")
        except Exception as e:
            raise Exception(f"AB-MCTS execution failed: {str(e)}")

    def _simple_multi_model(self, user_message: str) -> Generator[str, None, None]:
        """Fallback multi-model approach without TreeQuest"""
        
        yield "🤖 **Multi-Model Collaboration** (Simplified)\n\n"
        yield f"**Query:** {user_message}\n\n"

        responses = []
        
        # Get responses from all models
        for i, model in enumerate(self.valves.model_list):
            yield f"### Consulting {model}...\n"
            
            try:
                response = self._call_ollama(model, user_message)
                responses.append({"model": model, "response": response})
                yield f"**Response:** {response[:200]}{'...' if len(response) > 200 else ''}\n\n"
            except Exception as e:
                yield f"**Error:** {str(e)}\n\n"

        # Collaborative refinement
        if len(responses) > 1:
            yield "---\n\n## 🔄 **Collaborative Refinement**\n\n"
            
            for model_data in responses:
                other_responses = [r["response"] for r in responses if r["model"] != model_data["model"]]
                context = "\n".join([f"- {resp[:100]}..." for resp in other_responses])
                
                refinement_prompt = f"""Original question: {user_message}

Other models suggested:
{context}

Considering these perspectives, provide your refined response:"""

                try:
                    refined = self._call_ollama(model_data["model"], refinement_prompt)
                    yield f"### {model_data['model']} (Refined)\n"
                    yield f"{refined}\n\n"
                except Exception as e:
                    yield f"### {model_data['model']} (Refinement failed)\n"
                    yield f"Error: {str(e)}\n\n"

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
                f"{self.valves.ollama_base_url}/api/generate",
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
