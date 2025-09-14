"""
Search statistics model for tracking AB-MCTS performance.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SearchStats:
    """Statistics for AB-MCTS search performance."""
    
    total_iterations: int
    nodes_created: int
    best_reward: float
    average_reward: float
    exploration_ratio: float
    width_searches: int
    depth_searches: int
    model_usage: Dict[str, int]
    model_used: Optional[str] = None
    response_time: Optional[float] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_iterations": self.total_iterations,
            "nodes_created": self.nodes_created,
            "best_reward": self.best_reward,
            "average_reward": self.average_reward,
            "exploration_ratio": self.exploration_ratio,
            "width_searches": self.width_searches,
            "depth_searches": self.depth_searches,
            "model_usage": self.model_usage,
            "model_used": self.model_used,
            "response_time": self.response_time,
            "error_count": self.error_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchStats":
        """Create from dictionary."""
        return cls(
            total_iterations=data["total_iterations"],
            nodes_created=data["nodes_created"],
            best_reward=data["best_reward"],
            average_reward=data["average_reward"],
            exploration_ratio=data["exploration_ratio"],
            width_searches=data["width_searches"],
            depth_searches=data["depth_searches"],
            model_usage=data["model_usage"],
            model_used=data.get("model_used"),
            response_time=data.get("response_time"),
            error_count=data.get("error_count", 0)
        )
    
    @classmethod
    def empty(cls) -> "SearchStats":
        """Create empty stats."""
        return cls(
            total_iterations=0,
            nodes_created=0,
            best_reward=0.0,
            average_reward=0.0,
            exploration_ratio=0.0,
            width_searches=0,
            depth_searches=0,
            model_usage={},
            model_used=None,
            response_time=None,
            error_count=0
        )
