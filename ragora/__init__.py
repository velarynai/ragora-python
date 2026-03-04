"""
Ragora Python SDK

A simple, async-first wrapper for the Ragora API.
"""

from .client import (
    ChatAgenticOptions,
    ChatGenerationOptions,
    ChatMetadataOptions,
    ChatRetrievalOptions,
    RagoraClient,
    RequestOptions,
)
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
    AuthenticationError,
    AuthorizationError,
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
    NotFoundError,
    RagoraCitation,
    RagoraExtension,
    RagoraException,
    RateLimitError,
    SearchResponse,
    SearchResult,
    ServerError,
    ThinkingStep,
    UploadResponse,
)

from ._version import __version__  # noqa: F401
__all__ = [
    # Client
    "RagoraClient",
    "RequestOptions",
    "ChatGenerationOptions",
    "ChatRetrievalOptions",
    "ChatAgenticOptions",
    "ChatMetadataOptions",
    # Search
    "SearchResult",
    "SearchResponse",
    # Chat
    "ChatMessage",
    "ChatChoice",
    "ChatResponse",
    "ChatStreamChunk",
    "RagoraCitation",
    "RagoraExtension",
    "ThinkingStep",
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
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
]
