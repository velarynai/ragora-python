"""
Pydantic models for Ragora API responses.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """Metadata extracted from response headers."""
    
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    api_version: Optional[str] = Field(None, description="API version")
    cost_usd: Optional[float] = Field(None, description="Cost of this request in USD")
    balance_remaining_usd: Optional[float] = Field(None, description="Remaining balance in USD")
    rate_limit_limit: Optional[int] = Field(None, description="Rate limit per window")
    rate_limit_remaining: Optional[int] = Field(None, description="Remaining requests in window")
    rate_limit_reset: Optional[int] = Field(None, description="Seconds until rate limit resets")


# --- Search Models ---

class SearchResult(BaseModel):
    """A single search result."""
    
    id: str = Field(..., description="Document chunk ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    document_id: Optional[str] = Field(None, description="Parent document ID")
    collection_id: Optional[str] = Field(None, description="Collection ID")


class SearchResponse(ResponseMetadata):
    """Search API response."""
    
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    query: str = Field(..., description="Original query")
    total: int = Field(0, description="Total matching results")


# --- Chat Models ---

class ChatMessage(BaseModel):
    """A chat message."""
    
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatChoice(BaseModel):
    """A chat completion choice."""
    
    index: int = Field(0, description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped")


class ChatResponse(ResponseMetadata):
    """Chat completion response (OpenAI-compatible)."""
    
    id: str = Field(..., description="Completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatChoice] = Field(default_factory=list, description="Completion choices")
    usage: Optional[dict[str, int]] = Field(None, description="Token usage")
    
    # RAG-specific fields
    sources: list[SearchResult] = Field(default_factory=list, description="Source documents used")


class ChatStreamChunk(BaseModel):
    """A streaming chat chunk."""
    
    content: str = Field("", description="Content delta")
    finish_reason: Optional[str] = Field(None, description="Why generation stopped")
    sources: list[SearchResult] = Field(default_factory=list, description="Sources (only in final chunk)")


# --- Credit Models ---

class CreditBalance(ResponseMetadata):
    """Credit balance response."""
    
    balance_usd: float = Field(..., description="Current balance in USD")
    currency: str = Field("USD", description="Currency code")


# --- Collection Models ---

class Collection(BaseModel):
    """A document collection."""

    id: str = Field(..., description="Collection ID")
    name: str = Field(..., description="Collection name")
    slug: Optional[str] = Field(None, description="URL-friendly slug")
    description: Optional[str] = Field(None, description="Collection description")
    total_documents: int = Field(0, description="Number of documents")
    total_vectors: int = Field(0, description="Number of vectors")
    total_chunks: int = Field(0, description="Number of chunks")
    total_size_bytes: int = Field(0, description="Total size in bytes")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class CollectionList(ResponseMetadata):
    """List of collections response."""

    data: list[Collection] = Field(default_factory=list, description="Collections")
    total: int = Field(0, description="Total count")
    limit: int = Field(20, description="Page size")
    offset: int = Field(0, description="Page offset")
    has_more: bool = Field(False, description="More pages available")


# --- Document Models ---

class Document(BaseModel):
    """A document in a collection."""

    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status: pending, processing, completed, failed")
    mime_type: Optional[str] = Field(None, description="MIME type")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    vector_count: int = Field(0, description="Number of vectors generated")
    chunk_count: int = Field(0, description="Number of chunks generated")
    collection_id: Optional[str] = Field(None, description="Parent collection ID")
    progress_percent: Optional[int] = Field(None, description="Processing progress (0-100)")
    progress_stage: Optional[str] = Field(None, description="Current processing stage")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class DocumentStatus(BaseModel):
    """Document processing status response."""

    id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    filename: str = Field(..., description="Original filename")
    mime_type: Optional[str] = Field(None, description="MIME type")
    vector_count: int = Field(0, description="Number of vectors generated")
    chunk_count: int = Field(0, description="Number of chunks generated")
    progress_percent: Optional[int] = Field(None, description="Processing progress (0-100)")
    progress_stage: Optional[str] = Field(None, description="Current processing stage")
    eta_seconds: Optional[int] = Field(None, description="Estimated time remaining")
    has_transcript: bool = Field(False, description="Whether transcript is available")
    is_active: bool = Field(True, description="Whether this is the active version")
    version_number: int = Field(1, description="Version number")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class DocumentList(ResponseMetadata):
    """List of documents response."""

    data: list[Document] = Field(default_factory=list, description="Documents")
    total: int = Field(0, description="Total count")
    limit: int = Field(50, description="Page size")
    offset: int = Field(0, description="Page offset")
    has_more: bool = Field(False, description="More pages available")


class UploadResponse(ResponseMetadata):
    """Document upload response."""

    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Initial status (usually 'pending')")
    collection_id: str = Field(..., description="Collection the document was uploaded to")
    message: Optional[str] = Field(None, description="Status message")


class DeleteResponse(BaseModel):
    """Delete operation response."""

    message: str = Field(..., description="Status message")
    id: str = Field(..., description="Deleted resource ID")
    deleted_at: Optional[str] = Field(None, description="Deletion timestamp")


# --- Marketplace Models ---

class Listing(BaseModel):
    """A pricing listing for a marketplace product."""

    id: str = Field(..., description="Listing ID")
    product_id: str = Field(..., description="Product ID")
    seller_id: str = Field(..., description="Seller ID")
    type: str = Field(..., description="Listing type: subscription, one_time, usage_based, free")
    price_amount_usd: float = Field(0.0, description="Price in USD")
    price_interval: Optional[str] = Field(None, description="Billing interval: month, year, or null for one_time")
    price_per_retrieval_usd: Optional[float] = Field(None, description="Per-retrieval price for usage_based listings")
    is_active: bool = Field(True, description="Whether listing is active")
    buyer_count: Optional[int] = Field(None, description="Number of buyers")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class MarketplaceProduct(BaseModel):
    """A product listed on the marketplace."""

    id: str = Field(..., description="Product ID")
    collection_id: Optional[str] = Field(None, description="Linked collection ID")
    seller_id: str = Field(..., description="Seller user ID")
    slug: str = Field(..., description="URL-friendly slug")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    status: str = Field("active", description="Product status: draft, active, archived")
    average_rating: float = Field(0.0, description="Average user rating")
    review_count: int = Field(0, description="Number of reviews")
    total_vectors: int = Field(0, description="Number of vectors")
    total_chunks: int = Field(0, description="Number of chunks")
    access_count: int = Field(0, description="Number of users with access")
    seller: Optional[dict[str, Any]] = Field(None, description="Seller info")
    listings: Optional[list[dict[str, Any]]] = Field(None, description="Pricing listings")
    categories: Optional[list[dict[str, Any]]] = Field(None, description="Product categories")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class MarketplaceList(ResponseMetadata):
    """List of marketplace products response."""

    data: list[MarketplaceProduct] = Field(default_factory=list, description="Products")
    total: int = Field(0, description="Total count")
    limit: int = Field(20, description="Page size")
    offset: int = Field(0, description="Page offset")
    has_more: bool = Field(False, description="More pages available")


# --- Error Models ---

class APIErrorDetail(BaseModel):
    """Additional error detail."""
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    reason: Optional[str] = Field(None, description="Specific reason")


class APIError(BaseModel):
    """Structured API error response."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    details: list[APIErrorDetail] = Field(default_factory=list, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")


class RagoraException(Exception):
    """Exception raised for Ragora API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int,
        error: Optional[APIError] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error
        self.request_id = request_id
    
    def __str__(self) -> str:
        parts = [f"[{self.status_code}] {self.message}"]
        if self.request_id:
            parts.append(f"(Request ID: {self.request_id})")
        return " ".join(parts)
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if this is a rate limit error."""
        return self.status_code == 429
    
    @property
    def is_auth_error(self) -> bool:
        """Check if this is an authentication error."""
        return self.status_code in (401, 403)
    
    @property
    def is_retryable(self) -> bool:
        """Check if this error is worth retrying."""
        return self.status_code in (429, 500, 502, 503, 504)
