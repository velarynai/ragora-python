"""
Agent State Schema

Defines the state that flows through the LangGraph agent.
Uses TypedDict for LangGraph compatibility with Pydantic-style validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class AgentPhase(str, Enum):
    """Current phase of the agent's execution."""
    ANALYZING = "analyzing"       # Breaking down the query
    SEARCHING = "searching"       # Executing searches
    EVALUATING = "evaluating"     # Checking context sufficiency
    SYNTHESIZING = "synthesizing" # Generating final answer
    COMPLETE = "complete"         # Done


class MessageRole(str, Enum):
    """Message roles for chat history."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ============================================================================
# Data Models (Pydantic for validation)
# ============================================================================


class Message(BaseModel):
    """A chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A search result from the knowledge base."""
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: Optional[str] = None
    
    def to_context_string(self, index: int) -> str:
        """Format as a citable context string."""
        return f"[{index}] {self.content}"


class SearchCall(BaseModel):
    """Record of a search operation."""
    query: str
    results: list[SearchResult]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryAnalysis(BaseModel):
    """Analysis of the user's query."""
    original_query: str
    intent: str
    search_queries: list[str]
    complexity: Literal["simple", "moderate", "complex"]
    requires_multiple_searches: bool
    reasoning: str


class ContextEvaluation(BaseModel):
    """Evaluation of gathered context."""
    is_sufficient: bool
    confidence: float  # 0.0 to 1.0
    missing_information: Optional[str] = None
    suggested_query: Optional[str] = None
    reasoning: str


# ============================================================================
# LangGraph State (TypedDict for graph compatibility)
# ============================================================================


class AgentState(TypedDict):
    """
    The state that flows through the LangGraph agent.
    
    Uses TypedDict for LangGraph compatibility.
    The `messages` field uses the add_messages reducer to properly
    handle message accumulation.
    """
    
    # Core conversation state
    messages: Annotated[Sequence[Any], add_messages]  # LangChain message objects
    
    # Current user query being processed
    current_query: str
    
    # Agent execution phase
    phase: str  # AgentPhase value
    
    # Query analysis results
    query_analysis: Optional[dict[str, Any]]  # QueryAnalysis as dict
    
    # Search state
    search_queries: list[str]           # Queries to execute
    search_index: int                    # Current query index
    search_results: list[dict[str, Any]] # All SearchResult dicts
    search_history: list[dict[str, Any]] # All SearchCall dicts
    
    # Context evaluation
    context_evaluation: Optional[dict[str, Any]]  # ContextEvaluation as dict
    iteration_count: int                          # Number of search iterations
    max_iterations: int                           # Maximum allowed iterations
    
    # Final output
    final_answer: Optional[str]
    citations: list[dict[str, Any]]
    
    # Metadata
    session_id: str
    collection_id: str
    model_name: str
    created_at: str
    

def create_initial_state(
    query: str,
    session_id: str,
    collection_id: str,
    model_name: str = "gpt-5.3-codex",
    max_iterations: int = 3,
) -> AgentState:
    """Create the initial state for a new query."""
    return AgentState(
        messages=[],
        current_query=query,
        phase=AgentPhase.ANALYZING.value,
        query_analysis=None,
        search_queries=[],
        search_index=0,
        search_results=[],
        search_history=[],
        context_evaluation=None,
        iteration_count=0,
        max_iterations=max_iterations,
        final_answer=None,
        citations=[],
        session_id=session_id,
        collection_id=collection_id,
        model_name=model_name,
        created_at=datetime.utcnow().isoformat(),
    )


# ============================================================================
# State Helper Functions
# ============================================================================


def get_context_text(state: AgentState) -> str:
    """Format all search results as citable context."""
    if not state["search_results"]:
        return "No relevant context found."
    
    parts = []
    for i, result in enumerate(state["search_results"], 1):
        content = result.get("content", "")
        parts.append(f"[{i}] {content}")
    
    return "\n\n".join(parts)


def get_unique_results(state: AgentState) -> list[dict[str, Any]]:
    """Get deduplicated search results."""
    seen_ids = set()
    unique = []
    for result in state["search_results"]:
        result_id = result.get("id")
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            unique.append(result)
    return unique


def calculate_context_quality(state: AgentState) -> float:
    """Calculate average relevance score of gathered context."""
    results = state["search_results"]
    if not results:
        return 0.0
    
    scores = [r.get("score", 0.0) for r in results]
    return sum(scores) / len(scores)
