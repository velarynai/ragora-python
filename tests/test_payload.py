"""Tests for search payload construction - verify correct field mapping."""

from unittest.mock import AsyncMock

import pytest

from ragora.client import RagoraClient


@pytest.fixture
def mock_client():
    """Create a client with mocked HTTP layer."""
    client = RagoraClient.__new__(RagoraClient)
    client._base_url = "https://api.ragora.app"
    client._api_key = "test-key"
    client._timeout = 30.0
    client._max_retries = 0
    client._user_agent = "test"
    client._debug = False
    client._collection_ref_cache = {}
    client._product_ref_cache = {}
    client._resolver_cache_ttl = 300
    return client


@pytest.mark.asyncio
async def test_search_payload_field_mapping(mock_client):
    """Verify search() constructs correct snake_case payload."""
    captured_payload = {}

    async def mock_request(method, path, *, json_data=None, params=None, request_id=None, timeout=None):
        if json_data:
            captured_payload.update(json_data)
        return {
            "results": [],
            "query": "test",
            "total": 0,
        }, {}

    mock_client._request = mock_request
    mock_client._resolve_collection_ids = AsyncMock(return_value=["coll-1"])

    await mock_client.search(
        query="test query",
        collection_id="coll-1",
        top_k=10,
        filters={"category": {"$in": ["tech"]}},
        source_type=["upload"],
        source_name=["my-source"],
        version=["v1.0"],
        version_mode="latest",
        document_keys=["key-1"],
        custom_tags=["tag1"],
        domain=["legal"],
        domain_filter_mode="strict",
        enable_reranker=True,
        graph_filter={"entities": ["Alice"], "entity_type": "PERSON"},
        temporal_filter={"since": "2024-01-01", "recency_weight": 0.5},
    )

    assert captured_payload["query"] == "test query"
    assert captured_payload["top_k"] == 10
    assert captured_payload["collection_ids"] == ["coll-1"]
    assert captured_payload["filters"] == {"category": {"$in": ["tech"]}}
    assert captured_payload["source_type"] == ["upload"]
    assert captured_payload["source_name"] == ["my-source"]
    assert captured_payload["version"] == ["v1.0"]
    assert captured_payload["version_mode"] == "latest"
    assert captured_payload["document_keys"] == ["key-1"]
    assert captured_payload["custom_tags"] == ["tag1"]
    assert captured_payload["domain"] == ["legal"]
    assert captured_payload["domain_filter_mode"] == "strict"
    assert captured_payload["enable_reranker"] is True
    assert captured_payload["graph_filter"] == {"entities": ["Alice"], "entity_type": "PERSON"}
    assert captured_payload["temporal_filter"] == {"since": "2024-01-01", "recency_weight": 0.5}

    # threshold should NOT be in payload
    assert "threshold" not in captured_payload


@pytest.mark.asyncio
async def test_search_minimal_payload(mock_client):
    """Verify search() with minimal args only sends query and top_k."""
    captured_payload = {}

    async def mock_request(method, path, *, json_data=None, params=None, request_id=None, timeout=None):
        if json_data:
            captured_payload.update(json_data)
        return {"results": [], "query": "hello", "total": 0}, {}

    mock_client._request = mock_request
    mock_client._resolve_collection_ids = AsyncMock(return_value=None)

    await mock_client.search(query="hello")

    assert captured_payload == {"query": "hello", "top_k": 5}
