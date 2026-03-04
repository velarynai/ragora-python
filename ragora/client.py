"""
Ragora API Client

Async-first HTTP client for the Ragora API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from typing import Any, AsyncIterator, Optional, TypedDict

import httpx

from ._version import __version__
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
    RagoraException,
    RagoraExtension,
    RateLimitError,
    SearchResponse,
    SearchResult,
    ServerError,
    ThinkingStep,
    UploadResponse,
)

logger = logging.getLogger("ragora")


class RequestOptions(TypedDict, total=False):
    """Per-request options that can be passed to any API method."""
    request_id: str
    timeout: float


class ChatGenerationOptions(TypedDict, total=False):
    """Chat generation options."""
    model: str
    temperature: float
    max_tokens: int


class ChatRetrievalOptions(TypedDict, total=False):
    """Chat retrieval options."""
    collection_id: str | list[str]
    collection: str | list[str]
    product_ids: list[str]
    products: str | list[str]
    top_k: int
    source_type: list[str]
    source_name: list[str]
    version: list[str]
    version_mode: str
    document_keys: list[str]
    custom_tags: list[str]
    domain: list[str]
    domain_filter_mode: str
    filters: dict[str, Any]
    enable_reranker: bool
    graph_filter: dict[str, Any]
    temporal_filter: dict[str, Any]


class ChatAgenticOptions(TypedDict, total=False):
    """Chat agentic/session options."""
    mode: str
    system_prompt: str
    session: bool
    session_id: str


class ChatMetadataOptions(TypedDict, total=False):
    """Chat metadata options."""
    source: str
    installation_id: str
    channel_id: str
    requester_id: str


class RagoraClient:
    """
    Async client for the Ragora API.
    
    Example:
        client = RagoraClient(api_key="your-api-key")
        results = await client.search(collection_id="...", query="...")
    """
    
    DEFAULT_BASE_URL = "https://api.ragora.app"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 2
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.AsyncClient] = None,
        user_agent_suffix: Optional[str] = None,
    ):
        """
        Initialize the Ragora client.

        Args:
            api_key: Your Ragora API key (or set RAGORA_API_KEY env var)
            base_url: API base URL (or set RAGORA_BASE_URL env var; default: https://api.ragora.app)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for rate-limit (429) and server errors (5xx).
                Set to 0 to disable automatic retries. Default: 2.
            http_client: Optional custom httpx.AsyncClient
            user_agent_suffix: Optional suffix appended to the User-Agent header
        """
        if api_key is None:
            api_key = os.environ.get("RAGORA_API_KEY")
        if api_key is None:
            raise ValueError(
                "api_key must be provided or set via the RAGORA_API_KEY environment variable"
            )
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("RAGORA_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self._user_agent = f"ragora-python/{__version__}"
        if user_agent_suffix:
            self._user_agent = f"{self._user_agent} {user_agent_suffix}"
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = http_client
        self._owns_client = http_client is None
        self._resolver_cache_ttl_seconds = 300
        self._collection_ref_cache: dict[str, tuple[str, float]] = {}
        self._product_ref_cache: dict[str, tuple[str, float]] = {}
    
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
                    "User-Agent": self._user_agent,
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
    
    @staticmethod
    def _retry_delay(attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        If the server sent a Retry-After / X-RateLimit-Reset header, use that
        as the base delay (uncapped — the server knows best). Otherwise fall
        back to exponential backoff capped at 30 seconds.
        """
        if retry_after is not None and retry_after > 0:
            base = retry_after
        else:
            base = min(2 ** attempt, 30)  # 1, 2, 4, 8, 16, 30
        # Add jitter: 0.5x–1.0x of base
        return base * (0.5 + random.random() * 0.5)

    @staticmethod
    def _get_retry_after(response: httpx.Response) -> Optional[float]:
        """Extract retry delay from response headers."""
        # Prefer standard Retry-After header
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        # Fall back to X-RateLimit-Reset (seconds until reset)
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return float(reset)
            except ValueError:
                pass
        return None

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make an API request and return (data, metadata)."""
        client = await self._ensure_client()
        logger.debug("Request: %s %s", method, path)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            kwargs: dict[str, Any] = {
                "method": method,
                "url": path,
                "json": json_data,
                "params": params,
            }
            if request_id:
                kwargs["headers"] = {"X-Request-ID": request_id}
            if timeout is not None:
                kwargs["timeout"] = timeout
            response = await client.request(**kwargs)

            metadata = self._extract_metadata(response)
            logger.debug("Response: %s %s -> %s", method, path, response.status_code)

            if response.is_success:
                return response.json(), metadata

            if (
                response.status_code in self.RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                delay = self._retry_delay(
                    attempt, self._get_retry_after(response)
                )
                logger.debug("Retry attempt %d after %.1fs delay", attempt + 1, delay)
                await asyncio.sleep(delay)
                continue

            await self._handle_error(response, metadata.get("request_id"))

        # Should be unreachable, but satisfy the type checker
        raise last_exc or Exception("request failed")

    @staticmethod
    def _parse_search_results(raw_results: Any) -> list[SearchResult]:
        """Normalize API result chunks into SearchResult models."""
        if not isinstance(raw_results, list):
            return []

        results: list[SearchResult] = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue

            raw_id = raw.get("id", raw.get("chunk_id", ""))
            if raw_id is None:
                raw_id = ""

            content = raw.get("text", raw.get("content", ""))
            if content is None:
                content = ""

            score_raw = raw.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0

            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Extract source_url: prefer top-level field, fall back to metadata
            source_url = raw.get("source_url") or metadata.get("source_url") or None

            results.append(
                SearchResult(
                    id=str(raw_id),
                    content=str(content),
                    score=score,
                    source_url=source_url,
                    metadata=metadata,
                    document_id=raw.get("document_id"),
                    collection_id=raw.get("collection_id"),
                )
            )

        return results

    @classmethod
    def _extract_chat_sources(cls, payload: dict[str, Any]) -> list[SearchResult]:
        """Extract chat sources from Ragora's extended or legacy response shapes."""
        ragora_stats = payload.get("ragora_stats")
        if isinstance(ragora_stats, dict):
            nested_sources = ragora_stats.get("sources")
            if isinstance(nested_sources, list):
                return cls._parse_search_results(nested_sources)
        return cls._parse_search_results(payload.get("sources", []))

    async def _upload_file(
        self,
        path: str,
        file_content: bytes,
        filename: str,
        data: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Upload a file using multipart/form-data."""
        form_data = data or {}
        logger.debug("Request: POST %s (upload: %s)", path, filename)

        upload_headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self._user_agent,
        }
        if request_id:
            upload_headers["X-Request-ID"] = request_id

        for attempt in range(self.max_retries + 1):
            files = {"file": (filename, file_content)}

            # Create a new client without default Content-Type header for multipart
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout if timeout is not None else self.timeout,
                headers=upload_headers,
            ) as upload_client:
                response = await upload_client.post(
                    path,
                    files=files,
                    data=form_data,
                )

            metadata = self._extract_metadata(response)
            logger.debug("Response: POST %s (upload) -> %s", path, response.status_code)

            if response.is_success:
                return response.json(), metadata

            if (
                response.status_code in self.RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                delay = self._retry_delay(
                    attempt, self._get_retry_after(response)
                )
                logger.debug("Retry attempt %d after %.1fs delay", attempt + 1, delay)
                await asyncio.sleep(delay)
                continue

            await self._handle_error(response, metadata.get("request_id"))

        raise Exception("upload failed")
    
    @staticmethod
    def _build_error(
        message: str,
        status_code: int,
        error: Optional[APIError] = None,
        request_id: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> RagoraException:
        """Map HTTP status code to the appropriate error subclass."""
        if status_code == 401:
            return AuthenticationError(message, status_code, error, request_id)
        if status_code == 403:
            return AuthorizationError(message, status_code, error, request_id)
        if status_code == 404:
            return NotFoundError(message, status_code, error, request_id)
        if status_code == 429:
            return RateLimitError(message, status_code, error, request_id, retry_after)
        if status_code >= 500:
            return ServerError(message, status_code, error, request_id)
        return RagoraException(message, status_code, error, request_id)

    async def _handle_error(
        self,
        response: httpx.Response,
        request_id: Optional[str] = None,
    ) -> None:
        """Handle error responses."""
        retry_after = self._get_retry_after(response)
        try:
            data = response.json()
            if "error" in data:
                error_data = data["error"]
                if isinstance(error_data, dict):
                    error = APIError(**error_data)
                    raise self._build_error(
                        message=error.message,
                        status_code=response.status_code,
                        error=error,
                        request_id=request_id,
                        retry_after=retry_after,
                    )
                else:
                    raise self._build_error(
                        message=str(error_data),
                        status_code=response.status_code,
                        request_id=request_id,
                        retry_after=retry_after,
                    )
            raise self._build_error(
                message=data.get("message", response.text),
                status_code=response.status_code,
                request_id=request_id,
                retry_after=retry_after,
            )
        except json.JSONDecodeError:
            raise self._build_error(
                message=response.text or f"HTTP {response.status_code}",
                status_code=response.status_code,
                request_id=request_id,
                retry_after=retry_after,
            )

    def _normalize_identifier_key(self, value: str) -> str:
        return value.strip().lower()

    def _cache_get(self, cache: dict[str, tuple[str, float]], key: str) -> Optional[str]:
        now = time.time()
        cached = cache.get(key)
        if cached is None:
            return None
        resolved, expires_at = cached
        if expires_at <= now:
            cache.pop(key, None)
            return None
        return resolved

    def _cache_set(self, cache: dict[str, tuple[str, float]], key: str, resolved_id: str) -> None:
        cache[key] = (resolved_id, time.time() + self._resolver_cache_ttl_seconds)

    @staticmethod
    def _preview_id(raw_id: str) -> str:
        if len(raw_id) <= 10:
            return raw_id
        return f"{raw_id[:8]}..."

    def _raise_ambiguous_identifier(self, kind: str, identifier: str, candidates: list[str]) -> None:
        message = (
            f"Ambiguous {kind} '{identifier}'. "
            f"Matches: {', '.join(candidates[:5])}. "
            "Use slug or UUID for an exact match."
        )
        raise RagoraException(
            message=message,
            status_code=400,
            error=APIError(code="AMBIGUOUS_IDENTIFIER", message=message),
        )

    def _raise_identifier_not_found(self, kind: str, identifier: str) -> None:
        message = (
            f"{kind.capitalize()} '{identifier}' was not found in your accessible scope. "
            "Use list endpoints or pass slug/UUID."
        )
        raise RagoraException(
            message=message,
            status_code=404,
            error=APIError(code="IDENTIFIER_NOT_FOUND", message=message),
        )

    async def _list_accessible_collections_raw(self) -> list[dict[str, Any]]:
        limit = 100
        offset = 0
        all_items: list[dict[str, Any]] = []

        for _ in range(50):
            params = {"limit": limit, "offset": offset}
            data, _ = await self._request("GET", "/v1/collections", params=params)

            page = data.get("data", [])
            if isinstance(page, list):
                all_items.extend(item for item in page if isinstance(item, dict))

            has_more = bool(data.get("hasMore", data.get("has_more", False)))
            if not has_more:
                break
            offset += limit

        return all_items

    async def _list_accessible_products_raw(self) -> list[dict[str, Any]]:
        data, _ = await self._request("GET", "/v1/products/accessible")
        items = data.get("data", [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    async def resolve_collection(self, collection: str) -> str:
        """
        Resolve a collection reference (UUID, slug, or name) to a collection UUID.
        """
        ref = collection.strip()
        if ref == "":
            raise ValueError("collection cannot be empty")

        cache_key = self._normalize_identifier_key(ref)
        cached = self._cache_get(self._collection_ref_cache, cache_key)
        if cached is not None:
            return cached

        try:
            parsed = uuid.UUID(ref)
            resolved = str(parsed)
            self._cache_set(self._collection_ref_cache, cache_key, resolved)
            return resolved
        except ValueError:
            pass

        collections = await self._list_accessible_collections_raw()
        ref_lower = ref.lower()

        id_matches = [c for c in collections if str(c.get("id", "")) == ref]
        if len(id_matches) == 1:
            resolved = str(id_matches[0]["id"])
            self._cache_set(self._collection_ref_cache, cache_key, resolved)
            return resolved

        slug_matches = [
            c for c in collections
            if isinstance(c.get("slug"), str) and c["slug"].lower() == ref_lower
        ]
        if len(slug_matches) == 1:
            resolved = str(slug_matches[0]["id"])
            self._cache_set(self._collection_ref_cache, cache_key, resolved)
            return resolved

        name_matches = [
            c for c in collections
            if isinstance(c.get("name"), str) and c["name"].lower() == ref_lower
        ]
        if len(name_matches) > 1:
            candidates = [
                f"{m.get('name', '')} (slug={m.get('slug', '-')}, id={self._preview_id(str(m.get('id', '')))})"
                for m in name_matches
            ]
            self._raise_ambiguous_identifier("collection", ref, candidates)
        if len(name_matches) == 1:
            resolved = str(name_matches[0]["id"])
            self._cache_set(self._collection_ref_cache, cache_key, resolved)
            return resolved

        # Convenience fallback: allow product slug/title as collection reference.
        products = await self._list_accessible_products_raw()
        product_slug_matches = [
            p for p in products
            if isinstance(p.get("slug"), str) and p["slug"].lower() == ref_lower
        ]
        product_title_matches = [
            p for p in products
            if isinstance(p.get("title"), str) and p["title"].lower() == ref_lower
        ]
        product_collection_slug_matches = [
            p for p in products
            if isinstance(p.get("collection_slug"), str) and p["collection_slug"].lower() == ref_lower
        ]
        product_collection_name_matches = [
            p for p in products
            if isinstance(p.get("collection_name"), str) and p["collection_name"].lower() == ref_lower
        ]
        product_matches = (
            product_slug_matches
            or product_title_matches
            or product_collection_slug_matches
            or product_collection_name_matches
        )
        product_matches = [
            p for p in product_matches
            if isinstance(p.get("collection_id"), str) and p["collection_id"].strip() != ""
        ]
        if len(product_matches) > 1:
            candidates = [
                f"{m.get('title', '')} (slug={m.get('slug', '-')}, id={self._preview_id(str(m.get('id', '')))})"
                for m in product_matches
            ]
            self._raise_ambiguous_identifier("collection", ref, candidates)
        if len(product_matches) == 1:
            resolved = str(product_matches[0]["collection_id"])
            self._cache_set(self._collection_ref_cache, cache_key, resolved)
            return resolved

        self._raise_identifier_not_found("collection", ref)
        return ""

    async def resolve_product(self, product: str) -> str:
        """
        Resolve a product reference (UUID, slug, or title) to a product UUID.
        """
        ref = product.strip()
        if ref == "":
            raise ValueError("product cannot be empty")

        cache_key = self._normalize_identifier_key(ref)
        cached = self._cache_get(self._product_ref_cache, cache_key)
        if cached is not None:
            return cached

        try:
            parsed = uuid.UUID(ref)
            resolved = str(parsed)
            self._cache_set(self._product_ref_cache, cache_key, resolved)
            return resolved
        except ValueError:
            pass

        products = await self._list_accessible_products_raw()
        ref_lower = ref.lower()

        id_matches = [p for p in products if str(p.get("id", "")) == ref]
        if len(id_matches) == 1:
            resolved = str(id_matches[0]["id"])
            self._cache_set(self._product_ref_cache, cache_key, resolved)
            return resolved

        slug_matches = [
            p for p in products
            if isinstance(p.get("slug"), str) and p["slug"].lower() == ref_lower
        ]
        if len(slug_matches) == 1:
            resolved = str(slug_matches[0]["id"])
            self._cache_set(self._product_ref_cache, cache_key, resolved)
            return resolved

        title_matches = [
            p for p in products
            if isinstance(p.get("title"), str) and p["title"].lower() == ref_lower
        ]
        if len(title_matches) > 1:
            candidates = [
                f"{m.get('title', '')} (slug={m.get('slug', '-')}, id={self._preview_id(str(m.get('id', '')))})"
                for m in title_matches
            ]
            self._raise_ambiguous_identifier("product", ref, candidates)
        if len(title_matches) == 1:
            resolved = str(title_matches[0]["id"])
            self._cache_set(self._product_ref_cache, cache_key, resolved)
            return resolved

        self._raise_identifier_not_found("product", ref)
        return ""

    def _raise_conflicting_reference_inputs(self, preferred: str, legacy: str) -> None:
        raise ValueError(f"Pass either '{preferred}' or '{legacy}', not both.")

    def _normalize_reference_list(self, refs: str | list[str], label: str) -> list[str]:
        raw_values = [refs] if isinstance(refs, str) else refs
        normalized: list[str] = []
        for value in raw_values:
            if not isinstance(value, str):
                raise TypeError(f"{label} values must be strings.")
            cleaned = value.strip()
            if cleaned == "":
                raise ValueError(f"{label} cannot contain empty values.")
            normalized.append(cleaned)
        if not normalized:
            raise ValueError(f"{label} cannot be empty.")
        return normalized

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize_identifier_key(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value)
        return deduped

    async def _resolve_collection_ids(
        self,
        *,
        collection: Optional[str | list[str]],
        collection_id: Optional[str | list[str]],
    ) -> Optional[list[str]]:
        if collection is not None and collection_id is not None:
            self._raise_conflicting_reference_inputs("collection", "collection_id")

        if collection is None:
            if collection_id is None:
                return None
            return self._normalize_reference_list(collection_id, "collection_id")

        refs = self._normalize_reference_list(collection, "collection")
        resolved_ids: list[str] = []
        for ref in refs:
            resolved_ids.append(await self.resolve_collection(ref))
        return self._dedupe_preserve_order(resolved_ids)

    async def _resolve_single_collection_id(
        self,
        *,
        collection: Optional[str],
        collection_id: Optional[str | list[str]],
    ) -> Optional[str]:
        resolved_ids = await self._resolve_collection_ids(
            collection=collection,
            collection_id=collection_id,
        )
        if resolved_ids is None:
            return None
        if len(resolved_ids) != 1:
            raise ValueError("Exactly one collection must be provided for this operation.")
        return resolved_ids[0]

    async def _resolve_product_ids(
        self,
        *,
        products: Optional[str | list[str]],
        product_ids: Optional[list[str]],
    ) -> Optional[list[str]]:
        if products is not None and product_ids is not None:
            self._raise_conflicting_reference_inputs("products", "product_ids")

        if products is None:
            if product_ids is None:
                return None
            normalized_ids = self._normalize_reference_list(product_ids, "product_ids")
            return self._dedupe_preserve_order(normalized_ids)

        refs = self._normalize_reference_list(products, "products")
        resolved_ids: list[str] = []
        for ref in refs:
            resolved_ids.append(await self.resolve_product(ref))
        return self._dedupe_preserve_order(resolved_ids)
    
    # --- Search ---
    
    async def search(
        self,
        query: str,
        collection_id: Optional[str] = None,
        collection: Optional[str | list[str]] = None,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        source_type: Optional[list[str]] = None,
        source_name: Optional[list[str]] = None,
        version: Optional[list[str]] = None,
        version_mode: Optional[str] = None,
        document_keys: Optional[list[str]] = None,
        custom_tags: Optional[list[str]] = None,
        domain: Optional[list[str]] = None,
        domain_filter_mode: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
        graph_filter: Optional[dict[str, Any]] = None,
        temporal_filter: Optional[dict[str, Any]] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> SearchResponse:
        """
        Search for relevant documents.

        Args:
            query: Search query
            collection_id: Collection ID or slug (legacy parameter)
            collection: Collection UUID/slug/name (or list); can also be a product slug/title
            top_k: Number of results to return (default: 5)
            filters: Metadata filters (MongoDB-style operators)
            source_type: Filter by source type (e.g., ["upload", "html", "youtube"])
            source_name: Filter by source name
            version: Filter by document version tags
            version_mode: Version mode: "latest" or "all"
            document_keys: Filter by specific document keys
            custom_tags: Filter by custom tags (OR logic)
            domain: Filter by domain (e.g., ["legal", "medical", "software_docs"])
            domain_filter_mode: "preferred" (boost, default) or "strict" (filter)
            enable_reranker: Toggle reranker for result refinement (default: false)
            graph_filter: Knowledge graph filter (e.g., {"entities": ["john"], "entity_type": "PERSON"})
            temporal_filter: Temporal filter (e.g., {"since": "2024-01-01T00:00:00Z", "recency_weight": 0.5})
            request_options: Per-request options (request_id, timeout)

        Returns:
            SearchResponse with results and metadata
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        collection_ids = await self._resolve_collection_ids(
            collection=collection,
            collection_id=collection_id,
        )
        if collection_ids is not None:
            payload["collection_ids"] = collection_ids
        if filters is not None:
            payload["filters"] = filters
        if source_type is not None:
            payload["source_type"] = source_type
        if source_name is not None:
            payload["source_name"] = source_name
        if version is not None:
            payload["version"] = version
        if version_mode is not None:
            payload["version_mode"] = version_mode
        if document_keys is not None:
            payload["document_keys"] = document_keys
        if custom_tags is not None:
            payload["custom_tags"] = custom_tags
        if domain is not None:
            payload["domain"] = domain
        if domain_filter_mode is not None:
            payload["domain_filter_mode"] = domain_filter_mode
        if enable_reranker is not None:
            payload["enable_reranker"] = enable_reranker
        if graph_filter is not None:
            payload["graph_filter"] = graph_filter
        if temporal_filter is not None:
            payload["temporal_filter"] = temporal_filter
        
        data, metadata = await self._request(
            "POST", "/v1/retrieve", json_data=payload,
            request_id=_rid, timeout=_tout,
        )

        results = self._parse_search_results(data.get("results", []))

        fragments = data.get("fragments")
        if not isinstance(fragments, list):
            fragments = []

        knowledge_graph = data.get("knowledge_graph")
        if not isinstance(knowledge_graph, dict):
            knowledge_graph = None

        global_graph_context = data.get("global_graph_context")
        if not isinstance(global_graph_context, dict):
            global_graph_context = None

        graph_debug = data.get("graph_debug")
        if not isinstance(graph_debug, dict):
            graph_debug = None
        
        return SearchResponse(
            object=data.get("object"),
            results=results,
            fragments=fragments,
            system_instruction=data.get("system_instruction"),
            knowledge_graph=knowledge_graph,
            global_graph_context=global_graph_context,
            knowledge_graph_summary=data.get("knowledge_graph_summary"),
            graph_debug=graph_debug,
            query=query,
            total=len(results),
            **metadata,
        )
    
    # --- Chat ---

    async def _build_chat_payload(
        self,
        *,
        messages: list[dict[str, str]],
        generation: Optional[ChatGenerationOptions],
        retrieval: Optional[ChatRetrievalOptions],
        agentic: Optional[ChatAgenticOptions],
        metadata: Optional[ChatMetadataOptions],
        stream: bool,
    ) -> dict[str, Any]:
        generation_options = generation or {}
        retrieval_options = retrieval or {}
        agentic_options = agentic or {}

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "temperature": generation_options.get("temperature", 0.7),
        }

        model = generation_options.get("model")
        if model is not None:
            payload["model"] = model
        max_tokens = generation_options.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        collection_ids = await self._resolve_collection_ids(
            collection=retrieval_options.get("collection"),
            collection_id=retrieval_options.get("collection_id"),
        )
        if collection_ids is not None:
            payload["collection_ids"] = collection_ids

        resolved_product_ids = await self._resolve_product_ids(
            products=retrieval_options.get("products"),
            product_ids=retrieval_options.get("product_ids"),
        )
        if resolved_product_ids is not None:
            payload["product_ids"] = resolved_product_ids

        for field in (
            "top_k",
            "source_type",
            "source_name",
            "version",
            "version_mode",
            "document_keys",
            "custom_tags",
            "domain",
            "domain_filter_mode",
            "filters",
            "enable_reranker",
            "graph_filter",
            "temporal_filter",
        ):
            value = retrieval_options.get(field)
            if value is not None:
                payload[field] = value

        if metadata is not None:
            payload["metadata"] = metadata

        if "mode" in agentic_options and agentic_options.get("mode") is not None:
            payload["mode"] = agentic_options["mode"]
        if "system_prompt" in agentic_options and agentic_options.get("system_prompt") is not None:
            payload["system_prompt"] = agentic_options["system_prompt"]
        if "session" in agentic_options and agentic_options.get("session") is not None:
            payload["session"] = bool(agentic_options["session"])
        if "session_id" in agentic_options and agentic_options.get("session_id") is not None:
            payload["session_id"] = agentic_options["session_id"]

        return payload

    @staticmethod
    def _extract_stream_session_id(payload: dict[str, Any]) -> Optional[str]:
        ragora_stats = payload.get("ragora_stats")
        if isinstance(ragora_stats, dict):
            conversation_id = ragora_stats.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id.strip():
                return conversation_id

        ragora = payload.get("ragora")
        if isinstance(ragora, dict):
            session_id = ragora.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                return session_id

        return None

    async def chat(
        self,
        messages: list[dict[str, str]],
        generation: Optional[ChatGenerationOptions] = None,
        retrieval: Optional[ChatRetrievalOptions] = None,
        agentic: Optional[ChatAgenticOptions] = None,
        metadata: Optional[ChatMetadataOptions] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> ChatResponse:
        """
        Generate a chat completion with RAG context.

        Args:
            messages: Chat messages (role/content dicts)
            generation: Generation options (model/temperature/max_tokens)
            retrieval: Retrieval options (scope, filters, top_k)
            agentic: Agentic/session options (mode/system_prompt/session/session_id)
            metadata: Request metadata for analytics
            request_options: Per-request options (request_id, timeout)
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        payload = await self._build_chat_payload(
            messages=messages,
            generation=generation,
            retrieval=retrieval,
            agentic=agentic,
            metadata=metadata,
            stream=False,
        )

        data, response_metadata = await self._request(
            "POST", "/v1/chat/completions", json_data=payload,
            request_id=_rid, timeout=_tout,
        )

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
            if isinstance(c, dict)
        ]
        sources = self._extract_chat_sources(data)

        ragora_data = data.get("ragora")
        ragora_ext = None
        if isinstance(ragora_data, dict):
            citations: list[RagoraCitation] = []
            for c in ragora_data.get("citations", []):
                if not isinstance(c, dict):
                    continue
                ref_raw = c.get("ref", 0)
                try:
                    ref = int(ref_raw)
                except (TypeError, ValueError):
                    ref = 0
                score_raw = c.get("score", 0.0)
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0
                citations.append(
                    RagoraCitation(
                        ref=ref,
                        text=str(c.get("text", "")),
                        source=str(c.get("source", "")),
                        score=score,
                    )
                )
            steps = ragora_data.get("steps", [])
            ragora_ext = RagoraExtension(
                citations=citations,
                steps=steps if isinstance(steps, list) else [],
                session_id=ragora_data.get("session_id") if isinstance(ragora_data.get("session_id"), str) else None,
            )

        return ChatResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", payload.get("model")),
            choices=choices,
            usage=data.get("usage"),
            sources=sources,
            ragora=ragora_ext,
            **response_metadata,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        generation: Optional[ChatGenerationOptions] = None,
        retrieval: Optional[ChatRetrievalOptions] = None,
        agentic: Optional[ChatAgenticOptions] = None,
        metadata: Optional[ChatMetadataOptions] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        """
        Stream a chat completion with RAG context.

        Args:
            messages: Chat messages (role/content dicts)
            generation: Generation options (model/temperature/max_tokens)
            retrieval: Retrieval options (scope, filters, top_k)
            agentic: Agentic/session options (mode/system_prompt/session/session_id)
            metadata: Request metadata for analytics
            request_options: Per-request options (request_id, timeout)
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        client = await self._ensure_client()
        payload = await self._build_chat_payload(
            messages=messages,
            generation=generation,
            retrieval=retrieval,
            agentic=agentic,
            metadata=metadata,
            stream=True,
        )

        _stream_kwargs: dict[str, Any] = {
            "method": "POST",
            "url": "/v1/chat/completions",
            "json": payload,
        }
        if _rid:
            _stream_kwargs["headers"] = {"X-Request-ID": _rid}
        if _tout is not None:
            _stream_kwargs["timeout"] = _tout

        async with client.stream(**_stream_kwargs) as response:
            if not response.is_success:
                await response.aread()
                await self._handle_error(response)

            event_name = "message"
            data_lines: list[str] = []
            current_session_id: Optional[str] = None

            def parse_sse_event(
                current_event: str,
                current_data_lines: list[str],
            ) -> tuple[Optional[ChatStreamChunk], bool]:
                nonlocal current_session_id

                if not current_data_lines:
                    return None, False

                data_str = "\n".join(current_data_lines)
                if data_str == "[DONE]":
                    return None, True

                try:
                    event_payload = json.loads(data_str)
                except json.JSONDecodeError:
                    return None, False

                if not isinstance(event_payload, dict):
                    return None, False

                extracted_session_id = self._extract_stream_session_id(event_payload)
                if extracted_session_id is not None:
                    current_session_id = extracted_session_id

                ragora_stats = event_payload.get("ragora_stats")
                parsed_stats = ragora_stats if isinstance(ragora_stats, dict) else None

                if current_event in {"ragora_status", "ragora.step"}:
                    thinking = None
                    step_type = event_payload.get("type")
                    step_message = event_payload.get("message")
                    if isinstance(step_type, str) and isinstance(step_message, str):
                        thinking = ThinkingStep(
                            type=step_type,
                            message=step_message,
                            timestamp=event_payload.get("timestamp", 0),
                        )
                    return ChatStreamChunk(
                        content="",
                        finish_reason=None,
                        sources=[],
                        thinking=thinking,
                        session_id=current_session_id,
                        stats=parsed_stats,
                    ), False

                if current_event in {"ragora_metadata", "ragora_complete"}:
                    sources = self._extract_chat_sources(event_payload)
                    if not sources and current_session_id is None and current_event != "ragora_complete":
                        return None, False
                    return ChatStreamChunk(
                        content="",
                        finish_reason=None,
                        sources=sources,
                        session_id=current_session_id,
                        stats=parsed_stats,
                    ), False

                choices = event_payload.get("choices", [])
                choice = choices[0] if isinstance(choices, list) and choices else {}
                if not isinstance(choice, dict):
                    choice = {}

                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    delta = {}

                content = delta.get("content", "")
                if content is None:
                    content = ""

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None and not isinstance(finish_reason, str):
                    finish_reason = str(finish_reason)

                sources = self._extract_chat_sources(event_payload)
                if not content and finish_reason is None and not sources:
                    return None, False

                return ChatStreamChunk(
                    content=str(content),
                    finish_reason=finish_reason,
                    sources=sources,
                    session_id=current_session_id,
                    stats=parsed_stats,
                ), False

            async for line in response.aiter_lines():
                if line == "":
                    chunk, done = parse_sse_event(event_name, data_lines)
                    data_lines = []
                    event_name = "message"
                    if chunk is not None:
                        yield chunk
                    if done:
                        break
                    continue

                if line.startswith("event:"):
                    event_name = line[6:].strip() or "message"
                    continue

                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                chunk, _ = parse_sse_event(event_name, data_lines)
                if chunk is not None:
                    yield chunk
    
    # --- Credits ---
    
    async def get_balance(
        self,
        request_options: Optional[RequestOptions] = None,
    ) -> CreditBalance:
        """
        Get current credit balance.

        Args:
            request_options: Per-request options (request_id, timeout)

        Returns:
            CreditBalance with current balance
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "GET", "/v1/credits/balance",
            request_id=_rid, timeout=_tout,
        )
        
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
        request_options: Optional[RequestOptions] = None,
    ) -> CollectionList:
        """
        List your collections.

        Args:
            limit: Number of results per page (max 100)
            offset: Pagination offset
            search: Optional search query
            request_options: Per-request options (request_id, timeout)

        Returns:
            CollectionList with collections and pagination info
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        data, metadata = await self._request(
            "GET", "/v1/collections", params=params,
            request_id=_rid, timeout=_tout,
        )
        
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
    
    async def get_collection(
        self,
        collection_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> Collection:
        """
        Get a specific collection by ID or slug.

        Args:
            collection_id: Collection ID or slug
            request_options: Per-request options (request_id, timeout)

        Returns:
            Collection details
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "GET", f"/v1/collections/{collection_id}",
            request_id=_rid, timeout=_tout,
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

    async def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        slug: Optional[str] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Collection name
            description: Optional description
            slug: Optional URL-friendly slug (auto-generated if not provided)
            request_options: Per-request options (request_id, timeout)

        Returns:
            Created collection
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if slug is not None:
            payload["slug"] = slug

        data, _ = await self._request(
            "POST", "/v1/collections", json_data=payload,
            request_id=_rid, timeout=_tout,
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

    async def update_collection(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        slug: Optional[str] = None,
        capability_config: Optional[dict[str, Any]] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> Collection:
        """
        Update an existing collection.

        Args:
            collection_id: Collection ID or slug
            name: New name (optional)
            description: New description (optional)
            slug: New slug (optional)
            capability_config: MCP tool configuration (optional)
            request_options: Per-request options (request_id, timeout)

        Returns:
            Updated collection
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if slug is not None:
            payload["slug"] = slug
        if capability_config is not None:
            payload["capability_config"] = capability_config

        data, _ = await self._request(
            "PATCH", f"/v1/collections/{collection_id}", json_data=payload,
            request_id=_rid, timeout=_tout,
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

    async def delete_collection(
        self,
        collection_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> DeleteResponse:
        """
        Delete a collection and all its documents.

        Args:
            collection_id: Collection ID or slug
            request_options: Per-request options (request_id, timeout)

        Returns:
            Deletion confirmation
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "DELETE", f"/v1/collections/{collection_id}",
            request_id=_rid, timeout=_tout,
        )

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
        collection: Optional[str] = None,
        relative_path: Optional[str] = None,
        release_tag: Optional[str] = None,
        version: Optional[str] = None,
        effective_at: Optional[str] = None,
        document_time: Optional[str] = None,
        expires_at: Optional[str] = None,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
        custom_tags: Optional[list[str]] = None,
        domain: Optional[str] = None,
        scan_mode: Optional[str] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> UploadResponse:
        """
        Upload a document to a collection.

        Args:
            file_content: File content as bytes
            filename: Original filename
            collection_id: Target collection ID or slug (legacy parameter)
            collection: Target collection UUID/slug/name or product slug/title
            relative_path: Relative path for directory-style uploads
            release_tag: Release tag for versioned documents
            version: Document version string
            effective_at: When the document becomes effective (ISO 8601)
            document_time: Document timestamp for temporal search (ISO 8601)
            expires_at: When the document expires (ISO 8601)
            source_type: Source type (e.g., "sec_filing", "web_crawl")
            source_name: Source name (e.g., "sec-edgar")
            custom_tags: List of custom tags for filtering
            domain: Content domain (e.g., "financial", "legal")
            scan_mode: Scan mode for processing

        Returns:
            Upload response with document ID
        """
        import json

        resolved_collection_id = await self._resolve_single_collection_id(
            collection=collection,
            collection_id=collection_id,
        )

        form_data: dict[str, Any] = {}
        if resolved_collection_id is not None:
            form_data["collection_id"] = resolved_collection_id
        if relative_path is not None:
            form_data["relative_path"] = relative_path
        if release_tag is not None:
            form_data["release_tag"] = release_tag
        if version is not None:
            form_data["version"] = version
        if effective_at is not None:
            form_data["effective_at"] = effective_at
        if document_time is not None:
            form_data["document_time"] = document_time
        if expires_at is not None:
            form_data["expires_at"] = expires_at
        if source_type is not None:
            form_data["source_type"] = source_type
        if source_name is not None:
            form_data["source_name"] = source_name
        if custom_tags is not None:
            form_data["custom_tags"] = json.dumps(custom_tags)
        if domain is not None:
            form_data["domain"] = domain
        if scan_mode is not None:
            form_data["scan_mode"] = scan_mode

        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._upload_file(
            "/v1/documents",
            file_content=file_content,
            filename=filename,
            data=form_data,
            request_id=_rid,
            timeout=_tout,
        )

        return UploadResponse(
            id=data.get("id", ""),
            filename=data.get("filename", filename),
            status=data.get("status", "pending"),
            collection_id=data.get("collection_id", resolved_collection_id or ""),
            message=data.get("message"),
            **metadata,
        )

    async def upload_file(
        self,
        file_path: str,
        collection_id: Optional[str] = None,
        collection: Optional[str] = None,
        relative_path: Optional[str] = None,
        release_tag: Optional[str] = None,
        version: Optional[str] = None,
        effective_at: Optional[str] = None,
        document_time: Optional[str] = None,
        expires_at: Optional[str] = None,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
        custom_tags: Optional[list[str]] = None,
        domain: Optional[str] = None,
        scan_mode: Optional[str] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> UploadResponse:
        """
        Upload a file from disk to a collection.

        Args:
            file_path: Path to the file on disk
            collection_id: Target collection ID or slug (legacy parameter)
            collection: Target collection UUID/slug/name or product slug/title
            relative_path: Relative path for directory-style uploads
            release_tag: Release tag for versioned documents
            version: Document version string
            effective_at: When the document becomes effective (ISO 8601)
            document_time: Document timestamp for temporal search (ISO 8601)
            expires_at: When the document expires (ISO 8601)
            source_type: Source type (e.g., "sec_filing", "web_crawl")
            source_name: Source name (e.g., "sec-edgar")
            custom_tags: List of custom tags for filtering
            domain: Content domain (e.g., "financial", "legal")
            scan_mode: Scan mode for processing

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
            collection=collection,
            relative_path=relative_path,
            release_tag=release_tag,
            version=version,
            effective_at=effective_at,
            document_time=document_time,
            expires_at=expires_at,
            source_type=source_type,
            source_name=source_name,
            custom_tags=custom_tags,
            domain=domain,
            scan_mode=scan_mode,
            request_options=request_options,
        )

    async def get_document_status(
        self,
        document_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> DocumentStatus:
        """
        Get the processing status of a document.

        Args:
            document_id: Document ID
            request_options: Per-request options (request_id, timeout)

        Returns:
            Document status with progress information
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "GET", f"/v1/documents/{document_id}/status",
            request_id=_rid, timeout=_tout,
        )

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
        collection: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        request_options: Optional[RequestOptions] = None,
    ) -> DocumentList:
        """
        List documents in a collection.

        Args:
            collection_id: Collection ID or slug (legacy parameter)
            collection: Collection UUID/slug/name or product slug/title
            limit: Number of results per page (max 200)
            offset: Pagination offset

        Returns:
            DocumentList with documents and pagination info
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        resolved_collection_id = await self._resolve_single_collection_id(
            collection=collection,
            collection_id=collection_id,
        )
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if resolved_collection_id is not None:
            params["collection_id"] = resolved_collection_id

        data, metadata = await self._request(
            "GET", "/v1/documents", params=params,
            request_id=_rid, timeout=_tout,
        )

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

    async def delete_document(
        self,
        document_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> DeleteResponse:
        """
        Delete a document.

        Args:
            document_id: Document ID
            request_options: Per-request options (request_id, timeout)

        Returns:
            Deletion confirmation
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "DELETE", f"/v1/documents/{document_id}",
            request_id=_rid, timeout=_tout,
        )

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
        request_options: Optional[RequestOptions] = None,
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
            status = await self.get_document_status(document_id, request_options=request_options)

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
        request_options: Optional[RequestOptions] = None,
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

        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "GET", "/v1/marketplace", params=params,
            request_id=_rid, timeout=_tout,
        )

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

    async def get_marketplace_product(
        self,
        product_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> MarketplaceProduct:
        """
        Get a marketplace product by ID or slug.

        Args:
            product_id: Product ID or slug
            request_options: Per-request options (request_id, timeout)

        Returns:
            MarketplaceProduct details
        """
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "GET", f"/v1/marketplace/{product_id}",
            request_id=_rid, timeout=_tout,
        )

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

    # --- Agents ---

    @staticmethod
    def _map_agent(raw: dict[str, Any]) -> Agent:
        memory_config = raw.get("memory_config")
        if not isinstance(memory_config, dict):
            memory_config = {}

        retrieval_policy = raw.get("retrieval_policy")
        if not isinstance(retrieval_policy, dict):
            maybe_from_memory = memory_config.get("retrieval_policy")
            retrieval_policy = maybe_from_memory if isinstance(maybe_from_memory, dict) else None

        budget_config = raw.get("budget_config")
        if not isinstance(budget_config, dict):
            budget_config = {}

        return Agent(
            id=str(raw.get("id", "")),
            org_id=str(raw.get("org_id", "")),
            name=str(raw.get("name", "")),
            type=str(raw.get("type", "support")),
            system_prompt=str(raw.get("system_prompt", "")),
            collection_ids=[str(v) for v in raw.get("collection_ids", []) if isinstance(v, str)],
            memory_config=memory_config,
            retrieval_policy=retrieval_policy,
            budget_config=budget_config,
            status=str(raw.get("status", "active")),
            created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else None,
            updated_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        )

    @staticmethod
    def _map_agent_session(raw: dict[str, Any]) -> AgentSession:
        return AgentSession(
            id=str(raw.get("id", "")),
            agent_id=str(raw.get("agent_id", "")),
            org_id=str(raw.get("org_id", "")),
            source=str(raw.get("source", "")),
            source_key=raw.get("source_key") if isinstance(raw.get("source_key"), str) else None,
            visitor_id=raw.get("visitor_id") if isinstance(raw.get("visitor_id"), str) else None,
            status=str(raw.get("status", "open")),
            message_count=raw.get("message_count") if isinstance(raw.get("message_count"), int) else 0,
            created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else None,
            updated_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        )

    async def create_agent(
        self,
        name: str,
        collection_ids: list[str],
        type: str = "support",
        system_prompt: Optional[str] = None,
        memory_config: Optional[dict[str, Any]] = None,
        retrieval_policy: Optional[dict[str, Any]] = None,
        budget_config: Optional[dict[str, Any]] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> Agent:
        """Create a new agent."""
        payload: dict[str, Any] = {
            "name": name,
            "type": type,
            "collection_ids": collection_ids,
        }
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if memory_config is not None:
            payload["memory_config"] = memory_config
        if retrieval_policy is not None:
            payload["retrieval_policy"] = retrieval_policy
        if budget_config is not None:
            payload["budget_config"] = budget_config

        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "POST", "/v1/agents", json_data=payload,
            request_id=_rid, timeout=_tout,
        )
        return self._map_agent(data)

    async def list_agents(
        self,
        request_options: Optional[RequestOptions] = None,
    ) -> AgentList:
        """List all agents."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "GET", "/v1/agents",
            request_id=_rid, timeout=_tout,
        )

        raw_agents = data.get("agents", [])
        agents = [
            self._map_agent(agent)
            for agent in raw_agents
            if isinstance(agent, dict)
        ]
        return AgentList(agents=agents, **metadata)

    async def get_agent(
        self,
        agent_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> Agent:
        """Get an agent by ID."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "GET", f"/v1/agents/{agent_id}",
            request_id=_rid, timeout=_tout,
        )
        return self._map_agent(data)

    async def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        collection_ids: Optional[list[str]] = None,
        memory_config: Optional[dict[str, Any]] = None,
        retrieval_policy: Optional[dict[str, Any]] = None,
        budget_config: Optional[dict[str, Any]] = None,
        status: Optional[str] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> Agent:
        """Update an agent."""
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if collection_ids is not None:
            payload["collection_ids"] = collection_ids
        if memory_config is not None:
            payload["memory_config"] = memory_config
        if retrieval_policy is not None:
            payload["retrieval_policy"] = retrieval_policy
        if budget_config is not None:
            payload["budget_config"] = budget_config
        if status is not None:
            payload["status"] = status

        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "PATCH", f"/v1/agents/{agent_id}", json_data=payload,
            request_id=_rid, timeout=_tout,
        )
        return self._map_agent(data)

    async def delete_agent(
        self,
        agent_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> DeleteResponse:
        """Delete an agent."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "DELETE", f"/v1/agents/{agent_id}",
            request_id=_rid, timeout=_tout,
        )
        return DeleteResponse(
            message=data.get("message", "Agent deleted"),
            id=data.get("id", agent_id),
        )

    async def agent_chat(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        collection_ids: Optional[list[str]] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> AgentChatResponse:
        """Chat with an agent."""
        payload: dict[str, Any] = {
            "message": message,
            "stream": False,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if collection_ids is not None:
            payload["collection_ids"] = collection_ids

        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "POST", f"/v1/agents/{agent_id}/chat", json_data=payload,
            request_id=_rid, timeout=_tout,
        )

        return AgentChatResponse(
            message=str(data.get("message", "")),
            session_id=str(data.get("session_id", "")),
            citations=data.get("citations", []) if isinstance(data.get("citations"), list) else [],
            stats=data.get("stats") if isinstance(data.get("stats"), dict) else None,
            **metadata,
        )

    async def agent_chat_stream(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        collection_ids: Optional[list[str]] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> AsyncIterator[AgentChatStreamChunk]:
        """Stream a chat with an agent."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "message": message,
            "stream": True,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if collection_ids is not None:
            payload["collection_ids"] = collection_ids

        stream_kwargs: dict[str, Any] = {
            "method": "POST",
            "url": f"/v1/agents/{agent_id}/chat",
            "json": payload,
        }
        if _rid:
            stream_kwargs["headers"] = {"X-Request-ID": _rid}
        if _tout is not None:
            stream_kwargs["timeout"] = _tout

        async with client.stream(**stream_kwargs) as response:
            if not response.is_success:
                await response.aread()
                await self._handle_error(response)

            event_name = "message"
            data_lines: list[str] = []
            current_session_id: Optional[str] = None

            def parse_sse_event(
                current_event: str,
                current_data_lines: list[str],
            ) -> tuple[Optional[AgentChatStreamChunk], bool]:
                nonlocal current_session_id

                if not current_data_lines:
                    return None, False

                data_str = "\n".join(current_data_lines)
                if data_str == "[DONE]":
                    return None, True

                try:
                    event_payload = json.loads(data_str)
                except json.JSONDecodeError:
                    return None, False

                if not isinstance(event_payload, dict):
                    return None, False

                extracted_session_id = self._extract_stream_session_id(event_payload)
                if extracted_session_id is not None:
                    current_session_id = extracted_session_id

                ragora_stats = event_payload.get("ragora_stats")
                parsed_stats = ragora_stats if isinstance(ragora_stats, dict) else None

                if current_event in {"ragora_status", "ragora.step"}:
                    step_type = event_payload.get("type")
                    step_status = event_payload.get("status")
                    step_message = event_payload.get("message")
                    return AgentChatStreamChunk(
                        content="",
                        session_id=current_session_id,
                        sources=[],
                        stats=parsed_stats,
                        thinking=ThinkingStep(
                            type=step_type if isinstance(step_type, str) else "working",
                            message=(
                                step_status if isinstance(step_status, str)
                                else step_message if isinstance(step_message, str)
                                else "Working..."
                            ),
                            timestamp=int(time.time() * 1000),
                        ),
                        done=False,
                    ), False

                if current_event == "ragora_metadata":
                    sources = self._extract_chat_sources(event_payload)
                    if not sources and current_session_id is None:
                        return None, False
                    return AgentChatStreamChunk(
                        content="",
                        session_id=current_session_id,
                        sources=sources,
                        stats=parsed_stats,
                        done=False,
                    ), False

                if current_event == "ragora_complete":
                    return AgentChatStreamChunk(
                        content="",
                        session_id=current_session_id,
                        sources=self._extract_chat_sources(event_payload),
                        stats=parsed_stats,
                        done=True,
                    ), False

                choices = event_payload.get("choices", [])
                choice = choices[0] if isinstance(choices, list) and choices else {}
                if not isinstance(choice, dict):
                    choice = {}
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    delta = {}
                content = delta.get("content")
                if not isinstance(content, str) or content == "":
                    return None, False
                return AgentChatStreamChunk(
                    content=content,
                    session_id=current_session_id,
                    sources=[],
                    done=False,
                ), False

            async for line in response.aiter_lines():
                if line == "":
                    chunk, done = parse_sse_event(event_name, data_lines)
                    data_lines = []
                    event_name = "message"
                    if chunk is not None:
                        yield chunk
                    if done:
                        break
                    continue

                if line.startswith("event:"):
                    event_name = line[6:].strip() or "message"
                    continue

                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                chunk, _ = parse_sse_event(event_name, data_lines)
                if chunk is not None:
                    yield chunk

    async def list_agent_sessions(
        self,
        agent_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> AgentSessionList:
        """List sessions for an agent."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "GET", f"/v1/agents/{agent_id}/sessions",
            request_id=_rid, timeout=_tout,
        )

        raw_sessions = data.get("sessions", [])
        sessions = [
            self._map_agent_session(session)
            for session in raw_sessions
            if isinstance(session, dict)
        ]
        return AgentSessionList(
            sessions=sessions,
            total=data.get("total", len(sessions)) if isinstance(data.get("total"), int) else len(sessions),
            **metadata,
        )

    async def get_agent_session(
        self,
        agent_id: str,
        session_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> AgentSessionDetail:
        """Get an agent session with its messages."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, metadata = await self._request(
            "GET", f"/v1/agents/{agent_id}/sessions/{session_id}",
            request_id=_rid, timeout=_tout,
        )

        raw_session = data.get("session")
        session = self._map_agent_session(raw_session if isinstance(raw_session, dict) else {})
        raw_messages = data.get("messages", [])
        messages = [
            AgentMessage(
                id=str(m.get("id", "")),
                session_id=str(m.get("session_id", "")),
                role=str(m.get("role", "")),
                content=str(m.get("content", "")),
                latency_ms=m.get("latency_ms") if isinstance(m.get("latency_ms"), int) else None,
                cost_usd=float(m.get("cost_usd")) if isinstance(m.get("cost_usd"), (int, float)) else None,
                model=m.get("model") if isinstance(m.get("model"), str) else None,
                created_at=m.get("created_at") if isinstance(m.get("created_at"), str) else None,
            )
            for m in raw_messages
            if isinstance(m, dict)
        ]
        return AgentSessionDetail(
            session=session,
            messages=messages,
            **metadata,
        )

    async def delete_agent_session(
        self,
        agent_id: str,
        session_id: str,
        request_options: Optional[RequestOptions] = None,
    ) -> DeleteResponse:
        """Delete/resolve an agent session and clean up its memory."""
        _rid = request_options.get("request_id") if request_options else None
        _tout = request_options.get("timeout") if request_options else None
        data, _ = await self._request(
            "DELETE", f"/v1/agents/{agent_id}/sessions/{session_id}",
            request_id=_rid, timeout=_tout,
        )
        return DeleteResponse(
            message=data.get("status", "resolved"),
            id=session_id,
        )

    # --- Auto-Pagination Iterators ---

    async def list_collections_iter(
        self,
        limit: int = 20,
        search: Optional[str] = None,
        request_options: Optional[RequestOptions] = None,
    ) -> AsyncIterator[Collection]:
        """Iterate over all collections, automatically handling pagination."""
        offset = 0
        while True:
            page = await self.list_collections(
                limit=limit, offset=offset, search=search,
                request_options=request_options,
            )
            for item in page.data:
                yield item
            if not page.has_more:
                break
            offset += page.limit

    async def list_documents_iter(
        self,
        collection_id: Optional[str] = None,
        collection: Optional[str] = None,
        limit: int = 50,
        request_options: Optional[RequestOptions] = None,
    ) -> AsyncIterator[Document]:
        """Iterate over all documents, automatically handling pagination."""
        offset = 0
        while True:
            page = await self.list_documents(
                collection_id=collection_id, collection=collection,
                limit=limit, offset=offset,
                request_options=request_options,
            )
            for item in page.data:
                yield item
            if not page.has_more:
                break
            offset += page.limit

    async def list_marketplace_iter(
        self,
        limit: int = 20,
        search: Optional[str] = None,
        category: Optional[str] = None,
        trending: bool = False,
        request_options: Optional[RequestOptions] = None,
    ) -> AsyncIterator[MarketplaceProduct]:
        """Iterate over all marketplace products, automatically handling pagination."""
        offset = 0
        while True:
            page = await self.list_marketplace(
                limit=limit, offset=offset, search=search,
                category=category, trending=trending,
                request_options=request_options,
            )
            for item in page.data:
                yield item
            if not page.has_more:
                break
            offset += page.limit
