"""Tests for upload form data construction."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_upload_sends_metadata_fields(mock_client):
    """Verify upload_document sends all metadata fields in form data."""
    captured_data = {}

    async def mock_upload_file(path, *, file_content, filename, data, request_id=None, timeout=None):
        captured_data.update(data)
        return {"id": "doc-1", "filename": filename, "status": "pending", "collection_id": "coll-1"}, {}

    mock_client._upload_file = mock_upload_file
    mock_client._resolve_single_collection_id = AsyncMock(return_value="coll-1")

    await mock_client.upload_document(
        file_content=b"test content",
        filename="test.pdf",
        collection_id="coll-1",
        relative_path="docs/legal/",
        release_tag="v2.0",
        version="2.0.0",
        effective_at="2024-06-01T00:00:00Z",
        document_time="2024-06-01T12:00:00Z",
        expires_at="2025-06-01T00:00:00Z",
        source_type="sec_filing",
        source_name="sec-edgar",
        custom_tags=["10-K", "annual"],
        domain="financial",
        scan_mode="hi_res",
    )

    assert captured_data["collection_id"] == "coll-1"
    assert captured_data["relative_path"] == "docs/legal/"
    assert captured_data["release_tag"] == "v2.0"
    assert captured_data["version"] == "2.0.0"
    assert captured_data["effective_at"] == "2024-06-01T00:00:00Z"
    assert captured_data["document_time"] == "2024-06-01T12:00:00Z"
    assert captured_data["expires_at"] == "2025-06-01T00:00:00Z"
    assert captured_data["source_type"] == "sec_filing"
    assert captured_data["source_name"] == "sec-edgar"
    assert json.loads(captured_data["custom_tags"]) == ["10-K", "annual"]
    assert captured_data["domain"] == "financial"
    assert captured_data["scan_mode"] == "hi_res"


@pytest.mark.asyncio
async def test_upload_omits_unset_metadata(mock_client):
    """Verify upload_document omits metadata fields that aren't set."""
    captured_data = {}

    async def mock_upload_file(path, *, file_content, filename, data, request_id=None, timeout=None):
        captured_data.update(data)
        return {"id": "doc-1", "filename": filename, "status": "pending", "collection_id": "coll-1"}, {}

    mock_client._upload_file = mock_upload_file
    mock_client._resolve_single_collection_id = AsyncMock(return_value="coll-1")

    await mock_client.upload_document(
        file_content=b"test",
        filename="doc.txt",
        collection_id="coll-1",
    )

    assert "collection_id" in captured_data
    assert "relative_path" not in captured_data
    assert "release_tag" not in captured_data
    assert "custom_tags" not in captured_data


@pytest.mark.asyncio
async def test_upload_file_reads_from_disk(mock_client, tmp_path):
    """Verify upload_file reads from disk and calls upload_document."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    captured_args = {}

    async def mock_upload_document(*, file_content, filename, **kwargs):
        captured_args["file_content"] = file_content
        captured_args["filename"] = filename
        captured_args.update(kwargs)
        from ragora.models import UploadResponse
        return UploadResponse(id="doc-1", filename=filename, status="pending", collection_id="coll-1")

    mock_client.upload_document = mock_upload_document

    await mock_client.upload_file(
        file_path=str(test_file),
        collection_id="coll-1",
    )

    assert captured_args["filename"] == "test.txt"
    assert captured_args["file_content"] == b"hello world"
