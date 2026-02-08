"""
Example: Chat completion with RAG context

This example shows how to use the chat API with 
both non-streaming and streaming responses.
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
        # --- Non-streaming chat ---
        print("=== Non-streaming Chat ===\n")
        
        response = await client.chat(
            collection_id=collection_id,
            messages=[
                {"role": "user", "content": "What is RAG and how does it work?"}
            ],
            model="gpt-4o-mini",
            temperature=0.7,
        )
        
        print(f"Assistant: {response.choices[0].message.content}\n")
        
        if response.sources:
            print(f"--- Sources ({len(response.sources)}) ---")
            for source in response.sources:
                print(f"  - {source.content[:100]}... (score: {source.score:.3f})")
            print()
        
        print(f"Usage: {response.usage}")
        print(f"Request ID: {response.request_id}")
        print(f"Cost: ${response.cost_usd:.6f}" if response.cost_usd else "Cost: N/A")
        
        # --- Streaming chat ---
        print("\n\n=== Streaming Chat ===\n")
        
        print("Assistant: ", end="", flush=True)
        
        sources = []
        async for chunk in client.chat_stream(
            collection_id=collection_id,
            messages=[
                {"role": "user", "content": "Explain the benefits of using RAG over fine-tuning"}
            ],
            model="gpt-4o-mini",
            temperature=0.7,
        ):
            print(chunk.content, end="", flush=True)
            
            # Sources are included in the final chunk
            if chunk.sources:
                sources = chunk.sources
        
        print("\n")
        
        if sources:
            print(f"--- Sources ({len(sources)}) ---")
            for source in sources:
                print(f"  - {source.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
