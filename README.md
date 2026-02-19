# Ragora Python SDK

Official Python SDK for the [Ragora](https://ragora.app) RAG API. Build AI-powered knowledge bases with semantic search and chat completions.

[![PyPI version](https://badge.fury.io/py/ragora-sdk.svg)](https://pypi.org/project/ragora-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install ragora-sdk
```

Or with uv:

```bash
uv add ragora-sdk
```

## Release Smoke Check

Run a quick pre-release smoke check that:
- builds the package (if `build` is installed),
- validates dist metadata (if `twine` is installed),
- executes a curated set of examples against a mocked client (no network/API key required).

```bash
python -m ragora.smoke
```

Or via CLI entrypoint:

```bash
ragora-smoke
```

Useful options:

```bash
# skip build/twine phase
python -m ragora.smoke --skip-prepare

# run a subset of examples
python -m ragora.smoke --examples search.py,credits.py,listings.py
```

## Quick Start

```python
import asyncio
from ragora import RagoraClient

async def main():
    client = RagoraClient(api_key="your-api-key")

    # Create a collection
    collection = await client.create_collection(
        name="My Knowledge Base",
        description="Documentation and guides"
    )
    print(f"Created collection: {collection.id}")

    # Upload a document
    upload = await client.upload_file(
        file_path="./document.pdf",
        collection_id=collection.id
    )
    print(f"Uploaded: {upload.filename} (ID: {upload.id})")

    # Wait for processing to complete
    status = await client.wait_for_document(upload.id)
    print(f"Processing complete: {status.vector_count} vectors created")

    # Search the collection
    results = await client.search(
        collection_id=collection.id,
        query="How do I get started?",
        top_k=5
    )
    for result in results.results:
        print(f"Score: {result.score:.3f} - {result.content[:100]}...")

    # Chat with your knowledge base
    response = await client.chat(
        collection_id=collection.id,
        messages=[{"role": "user", "content": "Summarize the main concepts"}]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- **Async-first** - Built on `httpx` for high-performance async operations
- **Full type hints** - Pydantic models with complete type coverage
- **Streaming support** - Real-time chat completions with async iterators
- **Document management** - Upload, track progress, and manage documents
- **Collection CRUD** - Create, update, delete, and list collections
- **Cost tracking** - Monitor API costs per request
- **Rate limit handling** - Access rate limit info from response metadata

## API Reference

### Client Initialization

```python
from ragora import RagoraClient

# Basic usage
client = RagoraClient(api_key="your-api-key")

# With custom settings
client = RagoraClient(
    api_key="your-api-key",
    base_url="https://api.ragora.app",  # default
    timeout=30.0  # seconds
)

# Using as async context manager
async with RagoraClient(api_key="your-api-key") as client:
    results = await client.search(...)
```

### Collections

```python
# Create a collection
collection = await client.create_collection(
    name="My Collection",
    description="Optional description",
    slug="my-collection"  # optional, auto-generated if not provided
)

# List collections
collections = await client.list_collections(limit=20, offset=0)
for coll in collections.data:
    print(f"{coll.name}: {coll.total_documents} documents")

# Get a collection by ID or slug
collection = await client.get_collection("collection-id-or-slug")

# Update a collection
collection = await client.update_collection(
    "collection-id",
    name="New Name",
    description="Updated description"
)

# Delete a collection
result = await client.delete_collection("collection-id")
print(result.message)
```

### Documents

```python
# Upload from bytes
upload = await client.upload_document(
    file_content=b"Hello world",
    filename="hello.txt",
    collection_id="collection-id"  # optional, uses default if not provided
)

# Upload from file path
upload = await client.upload_file(
    file_path="./document.pdf",
    collection_id="collection-id"
)

# Check document status
status = await client.get_document_status(upload.id)
print(f"Status: {status.status}")
print(f"Progress: {status.progress_percent}%")
print(f"Stage: {status.progress_stage}")

# Wait for processing to complete
status = await client.wait_for_document(
    upload.id,
    timeout=300.0,      # max wait time in seconds
    poll_interval=2.0   # time between status checks
)

# List documents in a collection
documents = await client.list_documents(
    collection_id="collection-id",
    limit=50,
    offset=0
)

# Delete a document
result = await client.delete_document("document-id")
```

### Search

```python
results = await client.search(
    collection_id="collection-id",
    query="What is machine learning?",
    top_k=5,              # number of results
    threshold=0.7,        # minimum relevance score (0-1)
    filter={"type": "doc"}  # optional metadata filter
)

for result in results.results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}")
    print(f"Document ID: {result.document_id}")
    print("---")
```

### Chat Completions

```python
# Non-streaming
response = await client.chat(
    collection_id="collection-id",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG"}
    ],
    model="google/gemini-2.5-flash",    # optional
    temperature=0.7,         # optional
    max_tokens=1000,         # optional
    system_prompt="Custom system prompt"  # optional
)

print(response.choices[0].message.content)
print(f"Sources used: {len(response.sources)}")

# Streaming
async for chunk in client.chat_stream(
    collection_id="collection-id",
    messages=[{"role": "user", "content": "Explain RAG"}]
):
    print(chunk.content, end="", flush=True)

    # Sources are included in the final chunk
    if chunk.sources:
        print(f"\n\nSources: {len(chunk.sources)}")
```

### Marketplace

```python
# Browse marketplace products
products = await client.list_marketplace(limit=10, search="AI")
for product in products.data:
    print(f"{product.title} - {product.average_rating:.1f} stars")

# Get product details (by ID or slug)
product = await client.get_marketplace_product("product-slug")
print(f"{product.title}: {product.total_vectors} vectors")
if product.listings:
    for listing in product.listings:
        print(f"  {listing.get('type')}: ${listing.get('price_amount_usd', 0):.2f}")
```

### Credits

```python
balance = await client.get_balance()
print(f"Balance: ${balance.balance_usd:.2f} {balance.currency}")
```

## Response Metadata

Every response includes metadata from API headers:

```python
response = await client.search(...)

print(f"Request ID: {response.request_id}")
print(f"API Version: {response.api_version}")
print(f"Cost: ${response.cost_usd:.4f}")
print(f"Remaining balance: ${response.balance_remaining_usd:.2f}")
print(f"Rate limit: {response.rate_limit_remaining}/{response.rate_limit_limit}")
print(f"Rate limit resets in: {response.rate_limit_reset}s")
```

## Error Handling

```python
from ragora import RagoraClient, RagoraException

client = RagoraClient(api_key="your-api-key")

try:
    results = await client.search(...)
except RagoraException as e:
    print(f"Error: {e.message}")
    print(f"Status code: {e.status_code}")
    print(f"Request ID: {e.request_id}")

    if e.is_rate_limited:
        print("Rate limited - wait and retry")
    elif e.is_auth_error:
        print("Check your API key")
    elif e.is_retryable:
        print("Temporary error - safe to retry")
```

## Examples

See the [`examples/`](examples/) directory for complete, runnable examples:

| Example | Description |
|---------|-------------|
| [Search](examples/search.py) | Search documents and access response metadata |
| [Chat](examples/chat.py) | Chat completions with RAG context |
| [Streaming](examples/streaming.py) | Streaming chat responses |
| [Collections CRUD](examples/collections_crud.py) | Create, list, get, update, delete collections |
| [Documents](examples/documents.py) | Upload, process, list, delete documents |
| [Marketplace](examples/listings.py) | Browse marketplace products and listings |
| [Credits](examples/credits.py) | Check balance and track costs |

Set your API key before running:

```bash
export RAGORA_API_KEY="your-api-key"
python examples/search.py
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Ragora Website](https://ragora.app)
- [API Documentation](https://docs.ragora.app)
- [GitHub Repository](https://github.com/velarynai/ragora-python)
