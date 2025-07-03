#!/usr/bin/env python3
"""
ZUS Coffee Chatbot - Data Creation Script

This script scrapes ZUS Coffee data, creates vector embeddings, and stores everything
in the required formats for the Mindhive assessment.

Generated files:
- data/vectorstore.faiss (FAISS index for semantic search)
- data/vectorstore.pkl (metadata for products and stores)
- data/db.sqlite (structured outlet data for Text2SQL)

Usage:
    python create_data_locally.py [--force] [--data-dir DIR]
"""

import os
import sys
import sqlite3
import pickle
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add scrapers to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrapers.zus_scraper import scrape_zus_products, scrape_zus_stores

# Configuration
DEFAULT_DATA_DIR = "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def scrape_data() -> Tuple[List[Dict], List[Dict]]:
    """Scrape products and stores from ZUS Coffee websites"""
    logging.info("Scraping ZUS Coffee data...")

    # Scrape products
    logging.info("Scraping products from shop.zuscoffee.com...")
    products = scrape_zus_products()

    if not products:
        logging.warning("No products scraped, using fallback data")
        products = get_fallback_products()

    logging.info(f"Scraped {len(products)} products")

    # Scrape stores
    logging.info("Scraping store locations from zuscoffee.com...")
    stores = scrape_zus_stores()

    if not stores:
        logging.warning("No stores scraped, using fallback data")
        stores = get_fallback_stores()

    logging.info(f"Scraped {len(stores)} stores")

    return products, stores


def get_fallback_products() -> List[Dict]:
    """Fallback product data if scraping fails"""
    return [
        {
            "name": "ZUS Classic Tumbler",
            "price": ["RM 39.90"],
            "url": "https://shop.zuscoffee.com/products/classic-tumbler",
        },
        {
            "name": "ZUS Glass Coffee Cup",
            "price": ["RM 25.90"],
            "url": "https://shop.zuscoffee.com/products/glass-coffee-cup",
        },
        {
            "name": "ZUS Travel Mug",
            "price": ["RM 49.90"],
            "url": "https://shop.zuscoffee.com/products/travel-mug",
        },
        {
            "name": "ZUS Ceramic Mug Set",
            "price": ["RM 35.90"],
            "url": "https://shop.zuscoffee.com/products/ceramic-mug-set",
        },
        {
            "name": "ZUS Stainless Steel Water Bottle",
            "price": ["RM 42.90"],
            "url": "https://shop.zuscoffee.com/products/water-bottle",
        },
    ]


def get_fallback_stores() -> List[Dict]:
    """Fallback store data if scraping fails"""
    return [
        {
            "name": "ZUS Coffee KLCC",
            "location": "Kuala Lumpur City Centre, 50088 Kuala Lumpur",
        },
        {
            "name": "ZUS Coffee SS2",
            "location": "No. 72, Jalan SS 2/10, SS 2, 47300 Petaling Jaya, Selangor",
        },
        {
            "name": "ZUS Coffee Sunway Pyramid",
            "location": "3, Jalan PJS 11/15, Bandar Sunway, 47500 Selangor",
        },
    ]


def create_vectorstore(products: List[Dict], stores: List[Dict], data_dir: str) -> bool:
    """Create FAISS vectorstore with products and stores"""
    logging.info("Creating vectorstore...")

    try:
        # Load embedding model
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Prepare data for embedding
        all_items = []
        all_texts = []

        # Process products
        for product in products:
            # Extract price text
            if isinstance(product.get("price"), list):
                price_text = ", ".join(product["price"])
            else:
                price_text = str(product.get("price", ""))

            # Create comprehensive searchable text
            text_parts = [
                product["name"],
                f"Price: {price_text}",
                "ZUS Coffee drinkware product",
                "coffee accessories",
                "tumbler cup mug bottle",
            ]

            text = " ".join(part for part in text_parts if part.strip())
            all_texts.append(text)

            # Add item with only required fields and type marker
            item = {
                "name": product["name"],
                "price": product["price"],
                "url": product["url"],
                "_type": "product",
            }
            all_items.append(item)

        # Process stores
        for store in stores:
            text_parts = [
                f"ZUS Coffee outlet: {store['name']}",
                f"Location: {store['location']}",
                "ZUS Coffee branch store location",
                "coffee shop outlet",
            ]

            text = " ".join(text_parts)
            all_texts.append(text)

            # Add item with only required fields and type marker
            item = {
                "name": store["name"],
                "location": store["location"],
                "_type": "store",
            }
            all_items.append(item)

        logging.info(f"Creating embeddings for {len(all_items)} items...")
        embeddings = model.encode(all_texts, show_progress_bar=True)

        # Create FAISS index
        logging.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product similarity

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype("float32"))

        # Save files
        os.makedirs(data_dir, exist_ok=True)

        vectorstore_path = os.path.join(data_dir, "vectorstore.faiss")
        metadata_path = os.path.join(data_dir, "vectorstore.pkl")

        faiss.write_index(index, vectorstore_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(all_items, f)

        logging.info(f"Vectorstore saved: {vectorstore_path}")
        logging.info(f"Metadata saved: {metadata_path}")
        logging.info(
            f"Created embeddings for {len(products)} products and {len(stores)} stores"
        )

        return True

    except Exception as e:
        logging.error(f"Error creating vectorstore: {e}")
        return False


def create_sqlite_database(stores: List[Dict], data_dir: str) -> bool:
    """Create SQLite database with store data for Text2SQL"""
    logging.info("Creating SQLite database...")

    try:
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "db.sqlite")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create outlets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS outlets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location TEXT NOT NULL
            )
        """
        )

        # Clear existing data
        cursor.execute("DELETE FROM outlets")

        # Insert scraped store data
        for store in stores:
            cursor.execute(
                """
                INSERT INTO outlets (name, location)
                VALUES (?, ?)
            """,
                (
                    store["name"],
                    store["location"],
                ),
            )

        conn.commit()

        # Verify data
        cursor.execute("SELECT COUNT(*) FROM outlets")
        count = cursor.fetchone()[0]

        conn.close()

        logging.info(f"SQLite database created: {db_path}")
        logging.info(f"Inserted {count} outlets")

        return True

    except Exception as e:
        logging.error(f"Error creating SQLite database: {e}")
        return False


def check_existing_data(data_dir: str) -> bool:
    """Check if data files already exist"""
    required_files = [
        os.path.join(data_dir, "vectorstore.faiss"),
        os.path.join(data_dir, "vectorstore.pkl"),
        os.path.join(data_dir, "db.sqlite"),
    ]

    all_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            all_exist = False
            break

    if all_exist:
        logging.info("All data files already exist")
        for file_path in required_files:
            size = os.path.getsize(file_path)
            logging.info(f"✓ {file_path} ({size:,} bytes)")

    return all_exist


def verify_files(data_dir: str) -> bool:
    """Verify all required files were created successfully"""
    logging.info("Verifying created files...")

    required_files = [
        os.path.join(data_dir, "vectorstore.faiss"),
        os.path.join(data_dir, "vectorstore.pkl"),
        os.path.join(data_dir, "db.sqlite"),
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            logging.info(f"✓ {file_path} ({size:,} bytes)")
        else:
            logging.error(f"✗ {file_path} - MISSING")
            all_exist = False

    return all_exist


def main(force_refresh: bool = False, data_dir: str = DEFAULT_DATA_DIR) -> bool:
    """Main function to create all data files"""
    logging.info("ZUS Coffee Chatbot - Data Creation")
    logging.info("=" * 50)

    # Check existing data
    if not force_refresh and check_existing_data(data_dir):
        logging.info("Data files already exist. Use --force to recreate.")
        return True

    try:
        # Scrape fresh data
        products, stores = scrape_data()

        if not products and not stores:
            logging.error("Failed to scrape any data")
            return False

        # Create vectorstore
        if not create_vectorstore(products, stores, data_dir):
            logging.error("Failed to create vectorstore")
            return False

        # Create SQLite database
        if not create_sqlite_database(stores, data_dir):
            logging.error("Failed to create SQLite database")
            return False

        # Verify all files
        if not verify_files(data_dir):
            logging.error("File verification failed")
            return False

        logging.info("=" * 50)
        logging.info("✓ SUCCESS: All data files created successfully!")
        logging.info(
            f"Created data for {len(products)} products and {len(stores)} stores"
        )
        logging.info(f"Files saved to: {os.path.abspath(data_dir)}")
        logging.info("\nNext steps:")
        logging.info("1. Upload the entire 'data/' folder to your deployment")
        logging.info("2. Deploy to Vercel/hosting platform")

        return True

    except Exception as e:
        logging.error(f"Data creation failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ZUS Coffee chatbot data files")
    parser.add_argument(
        "--force", action="store_true", help="Force refresh even if data exists"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory to save data files (default: {DEFAULT_DATA_DIR})",
    )

    args = parser.parse_args()

    try:
        success = main(force_refresh=args.force, data_dir=args.data_dir)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logging.info("Cancelled by user")
        sys.exit(1)
