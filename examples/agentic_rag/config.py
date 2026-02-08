"""
Agent Configuration

Centralized configuration for the agentic RAG system.
Supports multiple LLM providers and customizable behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    """Configuration for the LLM."""
    
    provider: LLMProvider = LLMProvider.OPENROUTER
    model: str = "google/gemini-3-flash-preview"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Provider-specific settings
    api_key: Optional[str] = None  # Falls back to env vars
    base_url: Optional[str] = None  # For custom endpoints or Ollama
    
    
    def get_model_kwargs(self) -> dict[str, Any]:
        """Get kwargs for LangChain model initialization."""
        import os

        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if self.api_key:
            if self.provider in (LLMProvider.OPENAI, LLMProvider.OPENROUTER):
                # Keep both aliases for compatibility across langchain-openai versions.
                kwargs["api_key"] = self.api_key
                kwargs["openai_api_key"] = self.api_key
            elif self.provider == LLMProvider.ANTHROPIC:
                kwargs["anthropic_api_key"] = self.api_key
        elif self.provider == LLMProvider.OPENROUTER:
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key
                kwargs["openai_api_key"] = env_key

        if self.provider == LLMProvider.OPENROUTER and not self.base_url:
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
            kwargs["openai_api_base"] = "https://openrouter.ai/api/v1"
        elif self.provider in (LLMProvider.OPENAI, LLMProvider.OPENROUTER) and self.base_url:
            kwargs["base_url"] = self.base_url
            kwargs["openai_api_base"] = self.base_url
        elif self.provider == LLMProvider.OLLAMA and self.base_url:
            kwargs["base_url"] = self.base_url

        return kwargs


@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    
    top_k: int = 5                    # Results per search
    threshold: float = 0.5            # Minimum relevance score
    max_total_results: int = 15       # Maximum results to gather
    deduplicate: bool = True          # Remove duplicate results


@dataclass
class AgentConfig:
    """
    Complete configuration for the agentic RAG agent.
    
    Example:
        config = AgentConfig(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-sonnet"),
            collection_id="my-collection",
            max_iterations=3,
        )
    """
    
    # Ragora settings
    ragora_api_key: Optional[str] = None  # Falls back to RAGORA_API_KEY env var
    ragora_base_url: str = "https://api.ragora.app"
    collection_id: str = ""
    
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Search settings
    search: SearchConfig = field(default_factory=SearchConfig)
    
    # Agent behavior
    max_iterations: int = 3           # Maximum search refinement loops
    min_confidence: float = 0.7       # Minimum confidence to answer
    enable_query_decomposition: bool = True  # Break complex queries into sub-queries
    
    # Persistence
    persistence_enabled: bool = True
    persistence_path: str = ".agentic_rag_sessions.db"
    
    # Streaming
    streaming_enabled: bool = True
    
    # Verbosity
    verbose: bool = False
    
    def validate(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.collection_id:
            errors.append("collection_id is required")
        
        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        
        if not 0 <= self.min_confidence <= 1:
            errors.append("min_confidence must be between 0 and 1")
        
        if self.search.top_k < 1:
            errors.append("search.top_k must be at least 1")

        if not 0 <= self.search.threshold <= 1:
            errors.append("search.threshold must be between 0 and 1")
        
        return errors


# ============================================================================
# Preset Configurations
# ============================================================================


def get_openai_config(
    model: str = "gpt-5.3-codex",
    collection_id: str = "",
    **kwargs: Any,
) -> AgentConfig:
    """Create config for OpenAI models."""
    return AgentConfig(
        collection_id=collection_id,
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model=model,
            **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens", "api_key"]},
        ),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "api_key"]},
    )


def get_anthropic_config(
    model: str = "claude-opus-4.6",
    collection_id: str = "",
    **kwargs: Any,
) -> AgentConfig:
    """Create config for Anthropic models."""
    return AgentConfig(
        collection_id=collection_id,
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens", "api_key"]},
        ),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "api_key"]},
    )


def get_ollama_config(
    model: str = "llama-4-70b",
    base_url: str = "http://localhost:11434",
    collection_id: str = "",
    **kwargs: Any,
) -> AgentConfig:
    """Create config for local Ollama models."""
    return AgentConfig(
        collection_id=collection_id,
        llm=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens"]},
        ),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
    )
