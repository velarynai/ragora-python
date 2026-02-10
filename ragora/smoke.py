"""
Release smoke checks for the Ragora SDK.

Usage:
    python -m ragora.smoke
    ragora-smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

from ragora.models import (
    APIError,
    ChatChoice,
    ChatMessage,
    ChatResponse,
    ChatStreamChunk,
    CreditBalance,
    MarketplaceList,
    MarketplaceProduct,
    RagoraException,
    SearchResponse,
    SearchResult,
)


DEFAULT_EXAMPLES = [
    "search.py",
    "chat.py",
    "streaming.py",
    "credits.py",
    "listings.py",
    "error_handling.py",
]


class FakeRagoraClient:
    """Small fake client so examples can be exercised without network access."""

    def __init__(self, *args, **kwargs):
        self.collection_id = os.environ.get("RAGORA_COLLECTION_ID", "smoke-collection-id")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def search(self, collection_id: str, query: str, **kwargs) -> SearchResponse:
        if collection_id == "non-existent-collection":
            raise RagoraException(
                message="Collection not found",
                status_code=404,
                error=APIError(code="not_found", message="Collection not found"),
                request_id="smoke-404",
            )
        if collection_id != self.collection_id:
            raise RagoraException(
                message=f"Expected collection_id={self.collection_id}, got {collection_id}",
                status_code=400,
                error=APIError(code="invalid_collection_id", message="Invalid collection_id for smoke run"),
                request_id="smoke-400",
            )
        return SearchResponse(
            query=query,
            total=1,
            results=[
                SearchResult(
                    id="sr_1",
                    content="RAG combines retrieval and generation to ground answers in knowledge.",
                    score=0.92,
                )
            ],
            request_id="smoke-search",
            api_version="smoke",
            cost_usd=0.0001,
            balance_remaining_usd=9.99,
            rate_limit_limit=100,
            rate_limit_remaining=99,
            rate_limit_reset=60,
        )

    async def chat(self, collection_id: str, messages: list[dict], **kwargs) -> ChatResponse:
        if collection_id != self.collection_id:
            raise RagoraException(
                message=f"Expected collection_id={self.collection_id}, got {collection_id}",
                status_code=400,
                error=APIError(code="invalid_collection_id", message="Invalid collection_id for smoke run"),
                request_id="smoke-400",
            )
        _ = messages
        return ChatResponse(
            id="chat_1",
            created=1730000000,
            model="smoke-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="RAG retrieves relevant context, then generates a grounded response.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            sources=[
                SearchResult(
                    id="sr_2",
                    content="Retrieved snippet used for answer grounding.",
                    score=0.88,
                )
            ],
            request_id="smoke-chat",
            cost_usd=0.0002,
        )

    async def chat_stream(self, collection_id: str, messages: list[dict], **kwargs):
        if collection_id != self.collection_id:
            raise RagoraException(
                message=f"Expected collection_id={self.collection_id}, got {collection_id}",
                status_code=400,
                error=APIError(code="invalid_collection_id", message="Invalid collection_id for smoke run"),
                request_id="smoke-400",
            )
        _ = messages
        yield ChatStreamChunk(content="RAG ")
        yield ChatStreamChunk(content="works ")
        yield ChatStreamChunk(content="by retrieving context.")
        yield ChatStreamChunk(
            content="",
            finish_reason="stop",
            sources=[
                SearchResult(
                    id="sr_3",
                    content="Source returned in final stream chunk.",
                    score=0.9,
                )
            ],
        )

    async def get_balance(self) -> CreditBalance:
        return CreditBalance(balance_usd=12.34, currency="USD", request_id="smoke-balance")

    async def list_marketplace(self, **kwargs) -> MarketplaceList:
        _ = kwargs
        return MarketplaceList(
            total=1,
            data=[
                MarketplaceProduct(
                    id="prod_1",
                    seller_id="seller_1",
                    slug="sample-kb",
                    title="Sample Knowledge Base",
                    description="A sample marketplace product for smoke checks.",
                    average_rating=4.7,
                    review_count=12,
                )
            ],
        )

    async def get_marketplace_product(self, product_id: str) -> MarketplaceProduct:
        return MarketplaceProduct(
            id=product_id,
            seller_id="seller_1",
            slug="sample-kb",
            title="Sample Knowledge Base",
            description="Detailed product used in smoke checks.",
            status="active",
            average_rating=4.7,
            review_count=12,
            total_vectors=1234,
            total_chunks=567,
            access_count=45,
            seller={"name": "Ragora"},
            listings=[{"type": "free", "is_active": True}],
            categories=[{"name": "AI"}, {"name": "RAG"}],
        )


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_prepare_checks(root: Path) -> None:
    if importlib.util.find_spec("build") is None:
        print("Skipping build check: install `build` to enable (`pip install build`).")
    else:
        print("Running: python -m build")
        _run([sys.executable, "-m", "build"], cwd=root)

    if importlib.util.find_spec("twine") is None:
        print("Skipping twine check: install `twine` to enable (`pip install twine`).")
    else:
        dist_files = sorted((root / "dist").glob("*"))
        if not dist_files:
            print("Skipping twine check: no files in dist/ (build may have been skipped).")
            return
        print("Running: python -m twine check dist/*")
        _run([sys.executable, "-m", "twine", "check", *[str(p) for p in dist_files]], cwd=root)


def _run_examples_with_fake_client(root: Path, examples: list[str]) -> None:
    import ragora

    original_client = ragora.RagoraClient
    ragora.RagoraClient = FakeRagoraClient
    try:
        for name in examples:
            path = root / "examples" / name
            print(f"Running example (mocked): {path}")
            runpy.run_path(str(path), run_name="__main__")
    finally:
        ragora.RagoraClient = original_client


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDK smoke checks before release.")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip package prep checks (build + twine metadata check).",
    )
    parser.add_argument(
        "--examples",
        default=",".join(DEFAULT_EXAMPLES),
        help="Comma-separated example files from examples/ to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    root = Path(__file__).resolve().parent.parent
    examples = [x.strip() for x in args.examples.split(",") if x.strip()]
    os.environ.setdefault("RAGORA_COLLECTION_ID", "smoke-collection-id")

    if not args.skip_prepare:
        _run_prepare_checks(root)

    _run_examples_with_fake_client(root, examples)
    print("Smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
