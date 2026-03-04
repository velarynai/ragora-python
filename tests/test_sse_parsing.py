"""Tests for SSE stream parsing - mock SSE lines -> ChatStreamChunk."""

import json

import pytest

from ragora.models import ChatStreamChunk, ThinkingStep


def test_chat_stream_chunk_with_thinking_step():
    """Verify a ThinkingStep can be constructed from an SSE ragora.step event."""
    # Simulate the event_payload from an SSE ragora.step event
    event_payload = {
        "type": "searching",
        "message": "Looking up relevant documents...",
        "timestamp": 1700000000000,
    }

    # This is what the client code does:
    thinking = None
    step_type = event_payload.get("type")
    step_message = event_payload.get("message")
    if isinstance(step_type, str) and isinstance(step_message, str):
        thinking = ThinkingStep(
            type=step_type,
            message=step_message,
            timestamp=event_payload.get("timestamp", 0),
        )

    chunk = ChatStreamChunk(
        content="",
        finish_reason=None,
        sources=[],
        thinking=thinking,
        session_id=None,
        stats=None,
    )

    assert chunk.thinking is not None
    assert chunk.thinking.type == "searching"
    assert chunk.thinking.message == "Looking up relevant documents..."
    assert chunk.thinking.timestamp == 1700000000000


def test_chat_stream_chunk_content_delta():
    """Verify a normal content delta chunk."""
    chunk = ChatStreamChunk(content="Hello, world!")
    assert chunk.content == "Hello, world!"
    assert chunk.finish_reason is None
    assert chunk.thinking is None
    assert chunk.sources == []


def test_chat_stream_chunk_with_finish_reason():
    """Verify chunk with finish_reason and sources."""
    from ragora.models import SearchResult

    chunk = ChatStreamChunk(
        content="",
        finish_reason="stop",
        sources=[
            SearchResult(id="s1", content="source text", score=0.9)
        ],
    )
    assert chunk.finish_reason == "stop"
    assert len(chunk.sources) == 1
    assert chunk.sources[0].score == 0.9


def test_thinking_step_missing_fields_skipped():
    """When SSE payload lacks type or message, thinking should be None."""
    event_payload = {"status": "working"}

    step_type = event_payload.get("type")
    step_message = event_payload.get("message")
    thinking = None
    if isinstance(step_type, str) and isinstance(step_message, str):
        thinking = ThinkingStep(
            type=step_type,
            message=step_message,
            timestamp=event_payload.get("timestamp", 0),
        )

    assert thinking is None


def test_thinking_step_default_timestamp():
    """ThinkingStep should default timestamp to 0."""
    step = ThinkingStep(type="thinking", message="Processing...")
    assert step.timestamp == 0
