"""
Ragora Python SDK

A simple, async-first wrapper for the Ragora API.
"""

from .client import RagoraClient
from .models import (
    APIError,
    Agent,
    AgentChatResponse,
    AgentChatStreamChunk,
    AgentList,
    AgentMessage,
    AgentSession,
    AgentSessionDetail,
    AgentSessionList,
    ChatChoice,
    ChatMessage,
    ChatResponse,
    ChatStreamChunk,
    Collection,
    CollectionList,
    CreditBalance,
    DeleteResponse,
    Document,
    DocumentList,
    DocumentStatus,
    Listing,
    MarketplaceList,
    MarketplaceProduct,
    RagoraException,
    SearchResponse,
    SearchResult,
    UploadResponse,
)

__version__ = "0.1.2"
__all__ = [
    # Client
    "RagoraClient",
    # Search
    "SearchResult",
    "SearchResponse",
    # Chat
    "ChatMessage",
    "ChatChoice",
    "ChatResponse",
    "ChatStreamChunk",
    # Collections
    "Collection",
    "CollectionList",
    # Documents
    "Document",
    "DocumentList",
    "DocumentStatus",
    "UploadResponse",
    "DeleteResponse",
    # Marketplace
    "MarketplaceProduct",
    "MarketplaceList",
    "Listing",
    # Credits
    "CreditBalance",
    # Agents
    "Agent",
    "AgentList",
    "AgentChatResponse",
    "AgentChatStreamChunk",
    "AgentSession",
    "AgentSessionList",
    "AgentMessage",
    "AgentSessionDetail",
    # Errors
    "APIError",
    "RagoraException",
]
