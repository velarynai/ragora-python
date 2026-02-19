"""
Agentic RAG — powered by Ragora's agent chat system.

All RAG logic (knowledge search, memory management, compaction, tool calls)
is handled server-side. This example shows how to create an agent and chat
with it using minimal code.
"""

from .agent import AgenticRAGAgent

__all__ = ["AgenticRAGAgent"]
