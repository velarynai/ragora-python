"""
Example: Credit balance and cost tracking

This example shows how to check your credit balance and
track per-request costs from response metadata.
"""

import asyncio
import os

from ragora import RagoraClient


async def main():
    client = RagoraClient(
        api_key=os.environ.get("RAGORA_API_KEY", "your-api-key"),
        base_url=os.environ.get("RAGORA_BASE_URL", "https://api.ragora.app"),
    )

    async with client:
        # --- Check credit balance ---
        print("=== Credit Balance ===\n")

        balance = await client.get_balance()

        print(f"Balance: ${balance.balance_usd:.2f} {balance.currency}")
        print(f"Request ID: {balance.request_id}")

        # --- Perform a search and show the cost ---
        print("\n\n=== Search with Cost Tracking ===\n")

        collection_id = os.environ.get("RAGORA_COLLECTION_ID", "your-collection-id")

        results = await client.search(
            collection_id=collection_id,
            query="What is retrieval augmented generation?",
            top_k=5,
        )

        print(f"Found {results.total} results")
        print()

        # Cost info from response headers
        print("--- Cost Information ---")
        print(f"Cost: ${results.cost_usd:.6f}" if results.cost_usd else "Cost: N/A")
        print(f"Balance remaining: ${results.balance_remaining_usd:.2f}" if results.balance_remaining_usd else "Balance remaining: N/A")

        # Rate limit info from response headers
        print()
        print("--- Rate Limit Information ---")
        print(f"Requests remaining: {results.rate_limit_remaining}/{results.rate_limit_limit}")
        print(f"Resets in: {results.rate_limit_reset}s" if results.rate_limit_reset else "Reset: N/A")


if __name__ == "__main__":
    asyncio.run(main())
