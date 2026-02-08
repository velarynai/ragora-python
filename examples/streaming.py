"""
Example: Streaming chat completions

This example shows how to use streaming chat completions
to receive responses token-by-token in real time.
"""

import asyncio
import os

from ragora import RagoraClient


async def main():
    client = RagoraClient(
        api_key=os.environ.get("RAGORA_API_KEY", "your-api-key"),
        base_url=os.environ.get("RAGORA_BASE_URL", "https://api.ragora.app"),
    )

    collection_id = os.environ.get("RAGORA_COLLECTION_ID", "your-collection-id")

    async with client:
        print("=== Streaming Chat ===\n")

        print("Assistant: ", end="", flush=True)

        async for chunk in client.chat_stream(
            collection_id=collection_id,
            messages=[{"role": "user", "content": "Explain how RAG works in 3 sentences."}],
            temperature=0.7,
        ):
            # Print each token as it arrives
            print(chunk.content, end="", flush=True)

            # Sources are included in the final chunk
            if chunk.sources:
                print(f"\n\n--- Sources ({len(chunk.sources)}) ---")
                for source in chunk.sources:
                    print(f"  - Score: {source.score:.3f} | {source.content[:80]}...")

        print()  # Final newline


if __name__ == "__main__":
    asyncio.run(main())
