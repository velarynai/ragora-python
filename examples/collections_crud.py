"""
Example: Collection CRUD operations

This example shows how to create, list, get, update, and delete
collections using the Ragora API.
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
        # --- Create a collection ---
        print("=== Create Collection ===\n")

        collection = await client.create_collection(
            name="My Research Papers",
            description="A collection of ML research papers for RAG",
        )

        print(f"Created collection: {collection.name}")
        print(f"  ID: {collection.id}")
        print(f"  Slug: {collection.slug}")
        print(f"  Description: {collection.description}")
        print(f"  Created at: {collection.created_at}")

        collection_id = collection.id

        # --- List collections (with pagination) ---
        print("\n\n=== List Collections ===\n")

        collections = await client.list_collections(limit=10, offset=0)

        print(f"Total collections: {collections.total}")
        print(f"Page size: {collections.limit}, Offset: {collections.offset}")
        print(f"Has more: {collections.has_more}\n")

        for coll in collections.data:
            print(f"  - {coll.name} (id: {coll.id}, docs: {coll.total_documents})")

        # --- Get collection by ID ---
        print("\n\n=== Get Collection ===\n")

        fetched = await client.get_collection(collection_id)

        print(f"Name: {fetched.name}")
        print(f"Description: {fetched.description}")
        print(f"Documents: {fetched.total_documents}")
        print(f"Chunks: {fetched.total_chunks}")

        # --- Update collection ---
        print("\n\n=== Update Collection ===\n")

        updated = await client.update_collection(
            collection_id=collection_id,
            name="ML Research Papers (Updated)",
            description="Updated description with more detail",
        )

        print(f"Updated name: {updated.name}")
        print(f"Updated description: {updated.description}")
        print(f"Updated at: {updated.updated_at}")

        # --- Delete collection ---
        print("\n\n=== Delete Collection ===\n")

        result = await client.delete_collection(collection_id)

        print(f"Deleted: {result.message}")
        print(f"ID: {result.id}")


if __name__ == "__main__":
    asyncio.run(main())
