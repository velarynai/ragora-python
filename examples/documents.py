"""
Example: Document upload and management

This example shows how to upload documents, wait for processing,
check status, list documents, and clean up.
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
        # --- Create a collection for our documents ---
        print("=== Setup: Create Collection ===\n")

        collection = await client.create_collection(
            name="Document Upload Demo",
            description="Temporary collection for the upload example",
        )
        collection_id = collection.id
        print(f"Created collection: {collection.name} (id: {collection_id})")

        # --- Upload a document (using bytes) ---
        print("\n\n=== Upload Document ===\n")

        demo_content = b"""
        Retrieval Augmented Generation (RAG) is a technique that combines
        information retrieval with text generation. It works by first
        retrieving relevant documents from a knowledge base, then using
        those documents as context for a language model to generate
        accurate, grounded responses.

        Key benefits of RAG include:
        - Reduced hallucinations through grounded context
        - Up-to-date information without retraining
        - Source attribution for generated answers
        - Cost-effective compared to fine-tuning
        """

        upload = await client.upload_document(
            file_content=demo_content,
            filename="rag-overview.txt",
            collection_id=collection_id,
        )

        print(f"Uploaded: {upload.filename}")
        print(f"Document ID: {upload.id}")
        print(f"Status: {upload.status}")
        print(f"Request ID: {upload.request_id}")

        document_id = upload.id

        # --- Wait for processing to complete ---
        print("\n\n=== Wait for Processing ===\n")

        print("Waiting for document to be processed...")
        status = await client.wait_for_document(
            document_id=document_id,
            timeout=120.0,
            poll_interval=2.0,
        )

        print(f"Status: {status.status}")
        print(f"Chunks: {status.chunk_count}")
        print(f"Vectors: {status.vector_count}")

        # --- Check document status directly ---
        print("\n\n=== Check Document Status ===\n")

        status = await client.get_document_status(document_id)

        print(f"Document: {status.filename}")
        print(f"Status: {status.status}")
        print(f"MIME type: {status.mime_type}")
        print(f"Chunks: {status.chunk_count}")
        print(f"Vectors: {status.vector_count}")
        print(f"Has transcript: {status.has_transcript}")
        print(f"Version: {status.version_number}")

        # --- List documents in the collection ---
        print("\n\n=== List Documents ===\n")

        doc_list = await client.list_documents(
            collection_id=collection_id,
            limit=10,
        )

        print(f"Total documents: {doc_list.total}")
        for doc in doc_list.data:
            print(f"  - {doc.filename} (status: {doc.status}, chunks: {doc.chunk_count})")

        # --- Delete document ---
        print("\n\n=== Delete Document ===\n")

        result = await client.delete_document(document_id)
        print(f"Deleted document: {result.message}")

        # --- Clean up: delete the collection ---
        print("\n\n=== Cleanup: Delete Collection ===\n")

        result = await client.delete_collection(collection_id)
        print(f"Deleted collection: {result.message}")


if __name__ == "__main__":
    asyncio.run(main())
