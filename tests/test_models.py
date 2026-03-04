"""Tests for Pydantic models - verify they can be constructed from sample API JSON."""

from ragora.models import (
    AgentChatStreamChunk,
    ChatStreamChunk,
    Collection,
    MarketplaceProduct,
    SearchResponse,
    SearchResult,
    ThinkingStep,
)


def test_search_result_from_api_json():
    data = {
        "id": "chunk-1",
        "content": "RAG combines retrieval and generation.",
        "score": 0.92,
        "source_url": "https://example.com/doc",
        "metadata": {"filename": "doc.pdf"},
        "document_id": "doc-1",
        "collection_id": "coll-1",
    }
    result = SearchResult(**data)
    assert result.id == "chunk-1"
    assert result.score == 0.92
    assert result.source_url == "https://example.com/doc"
    assert result.document_id == "doc-1"


def test_search_response_from_api_json():
    data = {
        "object": "list",
        "results": [
            {"id": "c1", "content": "text", "score": 0.9}
        ],
        "query": "test",
        "total": 1,
    }
    resp = SearchResponse(**data)
    assert resp.query == "test"
    assert len(resp.results) == 1
    assert resp.results[0].score == 0.9


def test_collection_has_owner_id():
    data = {
        "id": "coll-1",
        "owner_id": "user-123",
        "name": "Test Collection",
        "total_documents": 5,
        "total_vectors": 100,
        "total_chunks": 50,
        "total_size_bytes": 1024,
    }
    coll = Collection(**data)
    assert coll.owner_id == "user-123"


def test_collection_owner_id_optional():
    data = {
        "id": "coll-1",
        "name": "Test",
    }
    coll = Collection(**data)
    assert coll.owner_id is None


def test_marketplace_product_new_fields():
    data = {
        "id": "prod-1",
        "seller_id": "seller-1",
        "slug": "test-kb",
        "title": "Test KB",
        "thumbnail_url": "https://example.com/thumb.png",
        "data_size": "1.5 GB",
        "is_trending": True,
        "is_verified": False,
        "average_rating": 4.5,
        "review_count": 10,
        "total_vectors": 100,
        "total_chunks": 50,
        "access_count": 5,
    }
    product = MarketplaceProduct(**data)
    assert product.thumbnail_url == "https://example.com/thumb.png"
    assert product.data_size == "1.5 GB"
    assert product.is_trending is True
    assert product.is_verified is False


def test_marketplace_product_new_fields_optional():
    data = {
        "id": "prod-1",
        "seller_id": "seller-1",
        "slug": "test",
        "title": "Test",
    }
    product = MarketplaceProduct(**data)
    assert product.thumbnail_url is None
    assert product.data_size is None
    assert product.is_trending is None
    assert product.is_verified is None


def test_thinking_step_model():
    step = ThinkingStep(
        type="searching",
        message="Looking up relevant documents...",
        timestamp=1700000000000,
    )
    assert step.type == "searching"
    assert step.message == "Looking up relevant documents..."
    assert step.timestamp == 1700000000000


def test_chat_stream_chunk_with_thinking_step():
    chunk = ChatStreamChunk(
        content="",
        thinking=ThinkingStep(
            type="thinking",
            message="Processing query...",
            timestamp=123,
        ),
    )
    assert chunk.thinking is not None
    assert chunk.thinking.type == "thinking"
    assert chunk.thinking.message == "Processing query..."


def test_chat_stream_chunk_without_thinking():
    chunk = ChatStreamChunk(content="hello")
    assert chunk.thinking is None


def test_agent_chat_stream_chunk_with_thinking_step():
    chunk = AgentChatStreamChunk(
        content="",
        thinking=ThinkingStep(
            type="working",
            message="Generating response...",
            timestamp=456,
        ),
    )
    assert chunk.thinking is not None
    assert chunk.thinking.type == "working"
