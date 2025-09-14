"""
LLM State model for tracking model responses and metadata.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMState:
    """Represents the state of an LLM response in the search tree."""
    
    answer: str
    model_used: str
    action_type: str  # "width" or "depth"
    score: float
    depth: int
    parent_state: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "model_used": self.model_used,
            "action_type": self.action_type,
            "score": self.score,
            "depth": self.depth,
            "parent_state": self.parent_state,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMState":
        """Create from dictionary."""
        return cls(
            answer=data["answer"],
            model_used=data["model_used"],
            action_type=data["action_type"],
            score=data["score"],
            depth=data["depth"],
            parent_state=data.get("parent_state"),
            metadata=data.get("metadata")
        )
