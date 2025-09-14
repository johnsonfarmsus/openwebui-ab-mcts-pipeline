"""
Query and response models for API communication.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid


class QueryRequest(BaseModel):
    """Request model for query processing."""
    
    query: str = Field(..., description="The user's question or request")
    iterations: int = Field(20, ge=1, le=100, description="Number of search iterations")
    max_depth: int = Field(5, ge=1, le=15, description="Maximum search depth")
    models: List[str] = Field(default_factory=list, description="List of models to use")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    previous_messages: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation messages")


class QueryResponse(BaseModel):
    """Response model for query processing."""
    
    result: str = Field(..., description="The generated response")
    success: bool = Field(..., description="Whether the request was successful")
    search_stats: Dict[str, Any] = Field(default_factory=dict, description="Search statistics")
    conversation_id: str = Field(default="", description="Conversation ID")
    turn_id: str = Field(default="", description="Turn ID")
    error: str = Field(default="", description="Error message if any")


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    
    turn_id: str
    query: str
    response: str
    timestamp: str
    search_stats: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "timestamp": self.timestamp,
            "search_stats": self.search_stats or {},
            "model_used": self.model_used
        }


@dataclass
class Conversation:
    """A conversation thread."""
    
    conversation_id: str
    turns: List[ConversationTurn]
    created_at: str
    last_updated: str
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(turn)
        self.last_updated = turn.timestamp
    
    def get_context(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        recent_turns = self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns
        return [
            {"role": "user", "content": turn.query} if i % 2 == 0 else {"role": "assistant", "content": turn.response}
            for i, turn in enumerate(recent_turns)
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
