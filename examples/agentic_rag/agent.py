"""
Agentic RAG Agent

Thin wrapper around the Ragora SDK's agent chat system.
All the heavy lifting (knowledge search, memory management, compaction,
tool calls) is handled server-side by the agent service.

Usage:
    agent = await AgenticRAGAgent.create(collection_id="my-collection")
    result = await agent.chat("What are the key design choices?")
    print(result["message"])
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional

from ragora import RagoraClient


class AgenticRAGAgent:
    """
    Agent backed by Ragora's agentic chat system.

    The server handles search, memory, compaction, and tool orchestration.
    This class manages the client lifecycle and session continuity.

    Example:
        agent = await AgenticRAGAgent.create(collection_id="my-collection")
        result = await agent.chat("What are the key features?")
        print(result["message"])

        # Streaming
        async for chunk in agent.chat_stream("Tell me more"):
            print(chunk["content"], end="", flush=True)
    """

    def __init__(
        self,
        client: RagoraClient,
        agent_id: str,
        session_id: Optional[str] = None,
    ):
        self.client = client
        self.agent_id = agent_id
        self.session_id = session_id

    @classmethod
    async def create(
        cls,
        collection_id: Optional[str] = None,
        name: str = "RAG Agent",
        system_prompt: Optional[str] = None,
        budget_config: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AgenticRAGAgent:
        """
        Create or connect to an agent.

        If agent_id is provided, connects to an existing agent.
        Otherwise, creates a new one linked to the given collection.

        Args:
            collection_id: Collection to link (required when creating)
            name: Agent name (used when creating)
            system_prompt: Custom system prompt
            budget_config: Budget/search config (e.g. {"top_k": 10})
            api_key: Ragora API key (falls back to RAGORA_API_KEY env var)
            base_url: API base URL (falls back to RAGORA_BASE_URL env var)
            agent_id: Existing agent ID to connect to
        """
        key = api_key or os.environ.get("RAGORA_API_KEY")
        if not key:
            raise ValueError(
                "RAGORA_API_KEY is required. Set it as an env var or pass api_key."
            )

        client = RagoraClient(
            api_key=key,
            base_url=base_url or os.environ.get(
                "RAGORA_BASE_URL", "https://api.ragora.app"
            ),
        )

        if agent_id:
            await client.get_agent(agent_id)
            return cls(client=client, agent_id=agent_id)

        if not collection_id:
            raise ValueError(
                "collection_id is required when creating a new agent"
            )

        agent = await client.create_agent(
            name=name,
            collection_ids=[collection_id],
            system_prompt=system_prompt,
            budget_config=budget_config,
        )
        return cls(client=client, agent_id=agent.id)

    async def chat(self, message: str) -> dict[str, Any]:
        """
        Send a message and get a response.

        Automatically tracks the session ID for multi-turn conversations.
        """
        response = await self.client.agent_chat(
            agent_id=self.agent_id,
            message=message,
            session_id=self.session_id,
        )
        if response.session_id:
            self.session_id = response.session_id

        return {
            "message": response.message,
            "session_id": response.session_id,
            "citations": response.citations,
            "stats": response.stats,
        }

    async def chat_stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """
        Stream a response token by token.

        Yields dicts with keys: content, session_id, stats, done.
        """
        async for chunk in self.client.agent_chat_stream(
            agent_id=self.agent_id,
            message=message,
            session_id=self.session_id,
        ):
            if chunk.session_id:
                self.session_id = chunk.session_id
            yield {
                "content": chunk.content,
                "session_id": chunk.session_id,
                "stats": chunk.stats,
                "done": chunk.done,
            }

    def new_session(self) -> None:
        """Start a new conversation session."""
        self.session_id = None

    async def list_sessions(self):
        """List all sessions for this agent."""
        return await self.client.list_agent_sessions(self.agent_id)

    async def get_session(self, session_id: str):
        """Get a session with its messages."""
        return await self.client.get_agent_session(self.agent_id, session_id)

    async def delete_session(self, session_id: str):
        """Delete a session and clean up its memory."""
        return await self.client.delete_agent_session(self.agent_id, session_id)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close()
