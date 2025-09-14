"""
Model configuration model for managing AI models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    
    id: str
    name: str
    endpoint: str
    parameters: Dict[str, Any]
    enabled: bool = True
    performance_score: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "endpoint": self.endpoint,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "performance_score": self.performance_score,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "error_count": self.error_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        last_used = None
        if data.get("last_used"):
            last_used = datetime.fromisoformat(data["last_used"])
        
        return cls(
            id=data["id"],
            name=data["name"],
            endpoint=data["endpoint"],
            parameters=data["parameters"],
            enabled=data.get("enabled", True),
            performance_score=data.get("performance_score", 0.0),
            last_used=last_used,
            usage_count=data.get("usage_count", 0),
            error_count=data.get("error_count", 0)
        )
    
    def update_usage(self, success: bool = True) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        if not success:
            self.error_count += 1
        self.last_used = datetime.now()
    
    def calculate_performance_score(self) -> float:
        """Calculate performance score based on usage and errors."""
        if self.usage_count == 0:
            return 0.0
        
        success_rate = (self.usage_count - self.error_count) / self.usage_count
        return success_rate
