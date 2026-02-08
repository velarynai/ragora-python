"""
Example: Search documents in a collection

This example shows how to search for relevant documents
and access response metadata.
"""

import asyncio
import os

from ragora import RagoraClient


async def main():
    # Initialize client
    client = RagoraClient(
        api_key=os.environ.get("RAGORA_API_KEY", "your-api-key"),
        base_url=os.environ.get("RAGORA_BASE_URL", "https://api.ragora.app"),
    )
    
    async with client:
        # Search for documents
        results = await client.search(
            collection_id=os.environ.get("RAGORA_COLLECTION_ID", "your-collection-id"),
            query="What is retrieval augmented generation?",
            top_k=5,
            threshold=0.7,  # Only return results with score >= 0.7
        )
        
        # Print results
        print(f"Found {results.total} results\n")
        
        for i, result in enumerate(results.results, 1):
            print(f"--- Result {i} (score: {result.score:.3f}) ---")
            print(result.content[:200] + "..." if len(result.content) > 200 else result.content)
            print()
        
        # Access metadata from response headers
        print("--- Response Metadata ---")
        print(f"Request ID: {results.request_id}")
        print(f"API Version: {results.api_version}")
        print(f"Cost: ${results.cost_usd:.6f}" if results.cost_usd else "Cost: N/A")
        print(f"Balance Remaining: ${results.balance_remaining_usd:.2f}" if results.balance_remaining_usd else "Balance: N/A")
        print(f"Rate Limit Remaining: {results.rate_limit_remaining}/{results.rate_limit_limit}")


if __name__ == "__main__":
    asyncio.run(main())
