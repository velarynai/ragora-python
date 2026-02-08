"""
Ragora API Client

Async-first HTTP client for the Ragora API.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from .models import (
    APIError,
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


class RagoraClient:
    """
    Async client for the Ragora API.
    
    Example:
        client = RagoraClient(api_key="your-api-key")
        results = await client.search(collection_id="...", query="...")
    """
    
    DEFAULT_BASE_URL = "https://api.ragora.app"
    DEFAULT_TIMEOUT = 30.0
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the Ragora client.
        
        Args:
            api_key: Your Ragora API key
            base_url: API base URL (default: https://api.ragora.app)
            timeout: Request timeout in seconds
            http_client: Optional custom httpx.AsyncClient
        """
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None
    
    async def __aenter__(self) -> "RagoraClient":
        await self._ensure_client()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self.close()
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "ragora-python/0.1.0",
                },
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and self._owns_client:
            await self._client.aclose()
            self._client = None
    
    def _extract_metadata(self, response: httpx.Response) -> dict[str, Any]:
        """Extract metadata from response headers."""
        headers = response.headers
        
        def safe_float(key: str) -> Optional[float]:
            val = headers.get(key)
            if val:
                try:
                    return float(val)
                except ValueError:
                    pass
            return None
        
        def safe_int(key: str) -> Optional[int]:
            val = headers.get(key)
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
            return None
        
        return {
            "request_id": headers.get("X-Request-ID"),
            "api_version": headers.get("X-Ragora-API-Version"),
            "cost_usd": safe_float("X-Ragora-Cost-USD"),
            "balance_remaining_usd": safe_float("X-Ragora-Balance-Remaining-USD"),
            "rate_limit_limit": safe_int("X-RateLimit-Limit"),
            "rate_limit_remaining": safe_int("X-RateLimit-Remaining"),
            "rate_limit_reset": safe_int("X-RateLimit-Reset"),
        }
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make an API request and return (data, metadata)."""
        client = await self._ensure_client()

        response = await client.request(
            method=method,
            url=path,
            json=json_data,
            params=params,
        )

        metadata = self._extract_metadata(response)

        if not response.is_success:
            await self._handle_error(response, metadata.get("request_id"))

        return response.json(), metadata

    async def _upload_file(
        self,
        path: str,
        file_content: bytes,
        filename: str,
        data: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Upload a file using multipart/form-data."""
        client = await self._ensure_client()

        files = {"file": (filename, file_content)}
        form_data = data or {}

        # Create a new client without default Content-Type header for multipart
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ragora-python/0.1.0",
            },
        ) as upload_client:
            response = await upload_client.post(
                path,
                files=files,
                data=form_data,
            )

        metadata = self._extract_metadata(response)

        if not response.is_success:
            await self._handle_error(response, metadata.get("request_id"))

        return response.json(), metadata
    
    async def _handle_error(
        self,
        response: httpx.Response,
        request_id: Optional[str] = None,
    ) -> None:
        """Handle error responses."""
        try:
            data = response.json()
            if "error" in data:
                error_data = data["error"]
                if isinstance(error_data, dict):
                    error = APIError(**error_data)
                    raise RagoraException(
                        message=error.message,
                        status_code=response.status_code,
                        error=error,
                        request_id=request_id,
                    )
                else:
                    raise RagoraException(
                        message=str(error_data),
                        status_code=response.status_code,
                        request_id=request_id,
                    )
            raise RagoraException(
                message=data.get("message", response.text),
                status_code=response.status_code,
                request_id=request_id,
            )
        except json.JSONDecodeError:
            raise RagoraException(
                message=response.text or f"HTTP {response.status_code}",
                status_code=response.status_code,
                request_id=request_id,
            )
    
    # --- Search ---
    
    async def search(
        self,
        query: str,
        collection_id: Optional[str] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Search for relevant documents.

        Args:
            query: Search query
            collection_id: Collection ID or slug (omit to search all accessible collections)
            top_k: Number of results to return (default: 5)
            threshold: Minimum relevance score (0-1)
            filters: Metadata filters (MongoDB-style operators)

        Returns:
            SearchResponse with results and metadata
        """
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        if collection_id is not None:
            payload["collection_ids"] = [collection_id]
        if threshold is not None:
            payload["threshold"] = threshold
        if filters is not None:
            payload["filters"] = filters
        
        data, metadata = await self._request("POST", "/v1/retrieve", json_data=payload)
        
        results = [
            SearchResult(
                id=r.get("id", ""),
                content=r.get("text", r.get("content", "")),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
                document_id=r.get("document_id"),
                collection_id=r.get("collection_id"),
            )
            for r in data.get("results", [])
        ]
        
        return SearchResponse(
            results=results,
            query=query,
            total=len(results),
            **metadata,
        )
    
    # --- Chat ---
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        collection_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """
        Generate a chat completion with RAG context.

        Args:
            messages: Chat messages (role/content dicts)
            collection_id: Collection ID or slug (omit to use all accessible collections)
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_k: Number of chunks to retrieve for context

        Returns:
            ChatResponse with completion and sources
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": False,
        }
        if collection_id is not None:
            payload["collection_ids"] = [collection_id]
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_k is not None:
            payload["top_k"] = top_k
        
        data, metadata = await self._request("POST", "/v1/chat/completions", json_data=payload)
        
        choices = [
            ChatChoice(
                index=c.get("index", 0),
                message=ChatMessage(
                    role=c.get("message", {}).get("role", "assistant"),
                    content=c.get("message", {}).get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            )
            for c in data.get("choices", [])
        ]
        
        sources = [
            SearchResult(
                id=s.get("id", ""),
                content=s.get("text", s.get("content", "")),
                score=s.get("score", 0.0),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("sources", [])
        ]

        return ChatResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", model),
            choices=choices,
            usage=data.get("usage"),
            sources=sources,
            **metadata,
        )
    
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        collection_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """
        Stream a chat completion with RAG context.

        Args:
            messages: Chat messages (role/content dicts)
            collection_id: Collection ID or slug (omit to use all accessible collections)
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_k: Number of chunks to retrieve for context

        Yields:
            ChatStreamChunk with content deltas
        """
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": True,
        }
        if collection_id is not None:
            payload["collection_ids"] = [collection_id]
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_k is not None:
            payload["top_k"] = top_k
        
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            if not response.is_success:
                # Read the full response for error handling
                await response.aread()
                await self._handle_error(response)
            
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                    
                    # Sources may be included in the final chunk
                    sources = []
                    if "sources" in data:
                        sources = [
                            SearchResult(
                                id=s.get("id", ""),
                                content=s.get("text", s.get("content", "")),
                                score=s.get("score", 0.0),
                                metadata=s.get("metadata", {}),
                            )
                            for s in data["sources"]
                        ]
                    
                    yield ChatStreamChunk(
                        content=content,
                        finish_reason=finish_reason,
                        sources=sources,
                    )
                except json.JSONDecodeError:
                    continue
    
    # --- Credits ---
    
    async def get_balance(self) -> CreditBalance:
        """
        Get current credit balance.
        
        Returns:
            CreditBalance with current balance
        """
        data, metadata = await self._request("GET", "/v1/credits/balance")
        
        return CreditBalance(
            balance_usd=data.get("balance_usd", 0.0),
            currency=data.get("currency", "USD"),
            **metadata,
        )
    
    # --- Collections ---
    
    async def list_collections(
        self,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> CollectionList:
        """
        List your collections.
        
        Args:
            limit: Number of results per page (max 100)
            offset: Pagination offset
            search: Optional search query
            
        Returns:
            CollectionList with collections and pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        
        data, metadata = await self._request("GET", "/v1/collections", params=params)
        
        collections = [
            Collection(
                id=c.get("id", ""),
                name=c.get("name", ""),
                slug=c.get("slug"),
                description=c.get("description"),
                total_documents=c.get("total_documents", 0),
                total_vectors=c.get("total_vectors", 0),
                total_chunks=c.get("total_chunks", 0),
                total_size_bytes=c.get("total_size_bytes", 0),
                created_at=c.get("created_at"),
                updated_at=c.get("updated_at"),
            )
            for c in data.get("data", [])
        ]

        return CollectionList(
            data=collections,
            total=data.get("total", 0),
            limit=data.get("limit", limit),
            offset=data.get("offset", offset),
            has_more=data.get("hasMore", False),
            **metadata,
        )
    
    async def get_collection(self, collection_id: str) -> Collection:
        """
        Get a specific collection by ID or slug.

        Args:
            collection_id: Collection ID or slug

        Returns:
            Collection details
        """
        data, _ = await self._request("GET", f"/v1/collections/{collection_id}")

        # Handle nested data structure
        coll_data = data.get("data", data)

        return Collection(
            id=coll_data.get("id", ""),
            name=coll_data.get("name", ""),
            slug=coll_data.get("slug"),
            description=coll_data.get("description"),
            total_documents=coll_data.get("total_documents", 0),
            total_vectors=coll_data.get("total_vectors", 0),
            total_chunks=coll_data.get("total_chunks", 0),
            total_size_bytes=coll_data.get("total_size_bytes", 0),
            created_at=coll_data.get("created_at"),
            updated_at=coll_data.get("updated_at"),
        )

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Collection name
            description: Optional description
            slug: Optional URL-friendly slug (auto-generated if not provided)

        Returns:
            Created collection
        """
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if slug is not None:
            payload["slug"] = slug

        data, _ = await self._request("POST", "/v1/collections", json_data=payload)

        # Handle nested data structure
        coll_data = data.get("data", data)

        return Collection(
            id=coll_data.get("id", ""),
            name=coll_data.get("name", ""),
            slug=coll_data.get("slug"),
            description=coll_data.get("description"),
            total_documents=coll_data.get("total_documents", 0),
            total_vectors=coll_data.get("total_vectors", 0),
            total_chunks=coll_data.get("total_chunks", 0),
            total_size_bytes=coll_data.get("total_size_bytes", 0),
            created_at=coll_data.get("created_at"),
            updated_at=coll_data.get("updated_at"),
        )

    async def update_collection(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> Collection:
        """
        Update an existing collection.

        Args:
            collection_id: Collection ID or slug
            name: New name (optional)
            description: New description (optional)
            slug: New slug (optional)

        Returns:
            Updated collection
        """
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if slug is not None:
            payload["slug"] = slug

        data, _ = await self._request(
            "PATCH", f"/v1/collections/{collection_id}", json_data=payload
        )

        # Handle nested data structure
        coll_data = data.get("data", data)

        return Collection(
            id=coll_data.get("id", ""),
            name=coll_data.get("name", ""),
            slug=coll_data.get("slug"),
            description=coll_data.get("description"),
            total_documents=coll_data.get("total_documents", 0),
            total_vectors=coll_data.get("total_vectors", 0),
            total_chunks=coll_data.get("total_chunks", 0),
            total_size_bytes=coll_data.get("total_size_bytes", 0),
            created_at=coll_data.get("created_at"),
            updated_at=coll_data.get("updated_at"),
        )

    async def delete_collection(self, collection_id: str) -> DeleteResponse:
        """
        Delete a collection and all its documents.

        Args:
            collection_id: Collection ID or slug

        Returns:
            Deletion confirmation
        """
        data, _ = await self._request("DELETE", f"/v1/collections/{collection_id}")

        return DeleteResponse(
            message=data.get("message", "Collection deleted"),
            id=data.get("id", collection_id),
            deleted_at=data.get("deleted_at"),
        )

    # --- Documents ---

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        collection_id: Optional[str] = None,
    ) -> UploadResponse:
        """
        Upload a document to a collection.

        Args:
            file_content: File content as bytes
            filename: Original filename
            collection_id: Target collection ID or slug (uses default if not provided)

        Returns:
            Upload response with document ID
        """
        form_data: dict[str, Any] = {}
        if collection_id is not None:
            form_data["collection_id"] = collection_id

        data, metadata = await self._upload_file(
            "/v1/documents",
            file_content=file_content,
            filename=filename,
            data=form_data,
        )

        return UploadResponse(
            id=data.get("id", ""),
            filename=data.get("filename", filename),
            status=data.get("status", "pending"),
            collection_id=data.get("collection_id", collection_id or ""),
            message=data.get("message"),
            **metadata,
        )

    async def upload_file(
        self,
        file_path: str,
        collection_id: Optional[str] = None,
    ) -> UploadResponse:
        """
        Upload a file from disk to a collection.

        Args:
            file_path: Path to the file on disk
            collection_id: Target collection ID or slug (uses default if not provided)

        Returns:
            Upload response with document ID
        """
        import os

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_content = f.read()

        return await self.upload_document(
            file_content=file_content,
            filename=filename,
            collection_id=collection_id,
        )

    async def get_document_status(self, document_id: str) -> DocumentStatus:
        """
        Get the processing status of a document.

        Args:
            document_id: Document ID

        Returns:
            Document status with progress information
        """
        data, _ = await self._request("GET", f"/v1/documents/{document_id}/status")

        return DocumentStatus(
            id=data.get("id", document_id),
            status=data.get("status", "unknown"),
            filename=data.get("filename", ""),
            mime_type=data.get("mime_type"),
            vector_count=data.get("vector_count", 0),
            chunk_count=data.get("chunk_count", 0),
            progress_percent=data.get("progress_percent"),
            progress_stage=data.get("progress_stage"),
            eta_seconds=data.get("eta_seconds"),
            has_transcript=data.get("has_transcript", False),
            is_active=data.get("is_active", True),
            version_number=data.get("version_number", 1),
            created_at=data.get("created_at"),
        )

    async def list_documents(
        self,
        collection_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentList:
        """
        List documents in a collection.

        Args:
            collection_id: Collection ID or slug (lists all if not provided)
            limit: Number of results per page (max 200)
            offset: Pagination offset

        Returns:
            DocumentList with documents and pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if collection_id:
            params["collection_id"] = collection_id

        data, metadata = await self._request("GET", "/v1/documents", params=params)

        documents = [
            Document(
                id=d.get("id", ""),
                filename=d.get("filename", ""),
                status=d.get("status", "unknown"),
                mime_type=d.get("mime_type"),
                size_bytes=d.get("file_size_bytes", d.get("size_bytes")),
                vector_count=d.get("vector_count", 0),
                chunk_count=d.get("chunk_count", 0),
                collection_id=d.get("collection_id"),
                progress_percent=d.get("progress_percent"),
                progress_stage=d.get("progress_stage"),
                error_message=d.get("error_message"),
                created_at=d.get("created_at"),
                updated_at=d.get("updated_at"),
            )
            for d in data.get("data", [])
        ]

        return DocumentList(
            data=documents,
            total=data.get("total", 0),
            limit=data.get("limit", limit),
            offset=data.get("offset", offset),
            has_more=data.get("has_more", data.get("hasMore", False)),
            **metadata,
        )

    async def delete_document(self, document_id: str) -> DeleteResponse:
        """
        Delete a document.

        Args:
            document_id: Document ID

        Returns:
            Deletion confirmation
        """
        data, _ = await self._request("DELETE", f"/v1/documents/{document_id}")

        return DeleteResponse(
            message=data.get("message", "Document deleted"),
            id=data.get("id", document_id),
            deleted_at=data.get("deleted_at"),
        )

    async def wait_for_document(
        self,
        document_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> DocumentStatus:
        """
        Wait for a document to finish processing.

        Args:
            document_id: Document ID
            timeout: Maximum time to wait in seconds (default: 300)
            poll_interval: Time between status checks in seconds (default: 2)

        Returns:
            Final document status

        Raises:
            RagoraException: If document processing fails or times out
        """
        import asyncio
        import time

        start_time = time.time()

        while True:
            status = await self.get_document_status(document_id)

            if status.status == "completed":
                return status

            if status.status == "failed":
                raise RagoraException(
                    message=f"Document processing failed: {status.progress_stage}",
                    status_code=500,
                    request_id=document_id,
                )

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise RagoraException(
                    message=f"Timeout waiting for document {document_id} to process",
                    status_code=408,
                    request_id=document_id,
                )

            await asyncio.sleep(poll_interval)

    # --- Marketplace ---

    async def list_marketplace(
        self,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
        category: Optional[str] = None,
        trending: bool = False,
    ) -> MarketplaceList:
        """
        List public marketplace products.

        Args:
            limit: Number of results per page (max 100)
            offset: Pagination offset
            search: Optional search query
            category: Optional category filter
            trending: Only show trending products

        Returns:
            MarketplaceList with products and pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        if category:
            params["category"] = category
        if trending:
            params["trending"] = "true"

        data, metadata = await self._request("GET", "/v1/marketplace", params=params)

        products = [
            MarketplaceProduct(
                id=p.get("id", ""),
                collection_id=p.get("collection_id"),
                seller_id=p.get("seller_id", ""),
                slug=p.get("slug", ""),
                title=p.get("title", ""),
                description=p.get("description"),
                status=p.get("status", "active"),
                average_rating=p.get("average_rating", 0.0),
                review_count=p.get("review_count", 0),
                total_vectors=p.get("total_vectors", 0),
                total_chunks=p.get("total_chunks", 0),
                access_count=p.get("access_count", 0),
                seller=p.get("seller"),
                listings=p.get("listings"),
                categories=p.get("categories"),
                created_at=p.get("created_at"),
                updated_at=p.get("updated_at"),
            )
            for p in data.get("data", [])
        ]

        return MarketplaceList(
            data=products,
            total=data.get("total", 0),
            limit=data.get("limit", limit),
            offset=data.get("offset", offset),
            has_more=data.get("hasMore", False),
            **metadata,
        )

    async def get_marketplace_product(self, product_id: str) -> MarketplaceProduct:
        """
        Get a marketplace product by ID or slug.

        Args:
            product_id: Product ID or slug

        Returns:
            MarketplaceProduct details
        """
        data, _ = await self._request("GET", f"/v1/marketplace/{product_id}")

        return MarketplaceProduct(
            id=data.get("id", ""),
            collection_id=data.get("collection_id"),
            seller_id=data.get("seller_id", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            description=data.get("description"),
            status=data.get("status", "active"),
            average_rating=data.get("average_rating", 0.0),
            review_count=data.get("review_count", 0),
            total_vectors=data.get("total_vectors", 0),
            total_chunks=data.get("total_chunks", 0),
            access_count=data.get("access_count", 0),
            seller=data.get("seller"),
            listings=data.get("listings"),
            categories=data.get("categories"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
