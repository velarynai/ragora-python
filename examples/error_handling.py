"""
Example: Error handling and retries

This example shows how to handle API errors gracefully,
including rate limits and authentication errors.
"""

import asyncio
import os
import random

from ragora import RagoraClient, RagoraException


async def search_with_retry(
    client: RagoraClient,
    collection_id: str,
    query: str,
    max_retries: int = 3,
):
    """Search with exponential backoff retry on transient errors."""
    
    for attempt in range(max_retries):
        try:
            return await client.search(
                collection_id=collection_id,
                query=query,
                top_k=5,
            )
        except RagoraException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            # Don't retry auth errors
            if e.is_auth_error:
                print("Authentication error - check your API key")
                raise
            
            # Check if worth retrying
            if not e.is_retryable:
                print(f"Non-retryable error (status {e.status_code})")
                raise
            
            # Calculate backoff with jitter
            if attempt < max_retries - 1:
                if e.is_rate_limited:
                    # Use rate limit reset time if available
                    wait_time = 5  # Default wait
                    print(f"Rate limited - waiting {wait_time}s before retry")
                else:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Transient error - waiting {wait_time:.1f}s before retry")
                
                await asyncio.sleep(wait_time)
    
    raise Exception("Max retries exceeded")


async def main():
    client = RagoraClient(
        api_key=os.environ.get("RAGORA_API_KEY", "your-api-key"),
        base_url=os.environ.get("RAGORA_BASE_URL", "https://api.ragora.app"),
    )
    
    collection_id = os.environ.get("RAGORA_COLLECTION_ID", "your-collection-id")
    
    async with client:
        # --- Basic error handling ---
        print("=== Basic Error Handling ===\n")
        
        try:
            results = await client.search(
                collection_id="non-existent-collection",
                query="test query",
            )
        except RagoraException as e:
            print(f"Error: {e}")
            print(f"Status code: {e.status_code}")
            print(f"Request ID: {e.request_id}")
            
            if e.error:
                print(f"Error code: {e.error.code}")
                print(f"Error message: {e.error.message}")
        
        # --- Retry with backoff ---
        print("\n\n=== Retry with Backoff ===\n")
        
        try:
            results = await search_with_retry(
                client=client,
                collection_id=collection_id,
                query="What is machine learning?",
                max_retries=3,
            )
            print(f"Success! Found {results.total} results")
        except RagoraException as e:
            print(f"All retries failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
