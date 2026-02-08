"""
Example: Browse marketplace listings

This example shows how to browse the public marketplace,
view product details with pricing listings, and search
for products by keyword.

Note: Marketplace browsing does not require owning a collection.
Any authenticated user can browse public marketplace listings.
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
        # --- List marketplace products ---
        print("=== Browse Marketplace ===\n")

        products = await client.list_marketplace(limit=10, offset=0)

        print(f"Total products: {products.total}")
        print(f"Showing: {len(products.data)} results\n")

        for product in products.data:
            print(f"  - {product.title}")
            print(f"    ID: {product.id}")
            print(f"    Rating: {product.average_rating:.1f} ({product.review_count} reviews)")
            if product.description:
                desc = product.description[:100]
                print(f"    Description: {desc}...")
            print()

        # --- Get details of a specific product ---
        if products.data:
            product_id = products.data[0].id

            print("\n=== Product Details ===\n")

            product = await client.get_marketplace_product(product_id)

            print(f"Title: {product.title}")
            print(f"Slug: {product.slug}")
            print(f"Description: {product.description}")
            print(f"Status: {product.status}")
            print(f"Vectors: {product.total_vectors}")
            print(f"Chunks: {product.total_chunks}")
            print(f"Access count: {product.access_count}")

            if product.seller:
                print(f"Seller: {product.seller.get('full_name') or product.seller.get('name', 'Unknown')}")

            if product.listings:
                print(f"\nPricing Listings ({len(product.listings)}):")
                for listing in product.listings:
                    listing_type = listing.get("type", "unknown")
                    print(f"  - Type: {listing_type}")
                    if listing_type == "usage_based":
                        print(f"    Per retrieval: ${listing.get('price_per_retrieval_usd', 0):.5f}")
                    elif listing_type in ("subscription", "one_time"):
                        price = listing.get("price_amount_usd", 0)
                        interval = listing.get("price_interval", "")
                        print(f"    Price: ${price:.2f}" + (f"/{interval}" if interval else ""))
                    elif listing_type == "free":
                        print(f"    Price: Free")
                    print(f"    Active: {listing.get('is_active', True)}")

            if product.categories:
                cats = [c.get("name", "") for c in product.categories]
                print(f"\nCategories: {', '.join(cats)}")
        else:
            print("No marketplace products found.")

        # --- Search marketplace with query ---
        print("\n\n=== Search Marketplace ===\n")

        search_results = await client.list_marketplace(
            search="machine learning",
            limit=5,
        )

        print(f"Search results for 'machine learning': {search_results.total}")
        for product in search_results.data:
            print(f"  - {product.title} (rating: {product.average_rating:.1f})")


if __name__ == "__main__":
    asyncio.run(main())
