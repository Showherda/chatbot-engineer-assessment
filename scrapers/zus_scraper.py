import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
import random
from typing import List, Dict, Set
import re
from urllib.parse import urljoin
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
PRODUCT_CONFIG = {
    "base_url": "https://shop.zuscoffee.com",
    "min_delay": 2.0,
    "max_delay": 5.0,
    "max_depth": 2,
    "max_urls": 10,
    "max_products_per_page": 50,
    "request_timeout": 15,
    "max_requests": 50,
}

STORE_CONFIG = {
    "base_url": "https://zuscoffee.com/category/store/kuala-lumpur-selangor",
    "min_delay": 2.0,
    "max_delay": 4.0,
    "max_pages": 22,
    "min_page": 1,
    "request_timeout": 15,
    "max_requests": 30,
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

# Global state for request tracking
_request_count = 0
_last_request_time = 0


def create_session() -> requests.Session:
    """Create a session with retry strategy and safety features"""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
    )

    return session


def respect_rate_limit(min_delay: float, max_delay: float):
    """Ensure we don't make requests too quickly"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    min_interval = random.uniform(min_delay, max_delay)

    if time_since_last < min_interval:
        sleep_time = min_interval - time_since_last
        logging.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)

    _last_request_time = time.time()


def check_request_limits(max_requests: int) -> bool:
    """Check if we've hit our safety limits"""
    global _request_count
    if _request_count >= max_requests:
        logging.warning(f"Hit maximum request limit ({max_requests})")
        return False
    return True


def get_page_content(
    url: str, session: requests.Session, config: dict, retries: int = 3
):
    """Get page content with retries and safety checks"""
    global _request_count

    if not check_request_limits(config["max_requests"]):
        logging.warning("Request limit reached, stopping")
        return None

    # Rotate user agent
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})

    # Respect rate limiting
    respect_rate_limit(config["min_delay"], config["max_delay"])

    for attempt in range(retries):
        try:
            logging.info(f"Fetching {url} (attempt {attempt + 1}/{retries})")

            _request_count += 1
            response = session.get(url, timeout=config["request_timeout"])
            response.raise_for_status()

            # Check response size (safety check)
            content_length = len(response.content)
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                logging.warning(f"Response too large: {content_length} bytes")
                return None

            logging.info(f"Successfully fetched {url} ({content_length} bytes)")
            return response.content

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout for {url} on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request error for {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error fetching {url}: {e}")

        if attempt < retries - 1:
            backoff_time = random.uniform(2, 5) * (2**attempt)
            logging.info(f"Waiting {backoff_time:.2f}s before retry...")
            time.sleep(backoff_time)

    logging.error(f"Failed to fetch {url} after {retries} attempts")
    return None


def parse_sitemap(
    session: requests.Session,
    sitemap_url: str,
    config: dict,
    visited_sitemaps: Set[str],
    depth: int = 0,
) -> Set[str]:
    """Recursively parse sitemap to find drinkware URLs with safety limits"""
    if depth > config["max_depth"]:
        logging.info(f"Max depth ({config['max_depth']}) reached, stopping recursion")
        return set()

    if sitemap_url in visited_sitemaps:
        logging.info(f"Already visited {sitemap_url}, skipping")
        return set()

    if not check_request_limits(config["max_requests"]):
        return set()

    visited_sitemaps.add(sitemap_url)
    logging.info(f"Parsing sitemap (depth {depth}): {sitemap_url}")

    try:
        content = get_page_content(sitemap_url, session, config)
        if not content:
            return set()

        # Parse XML safely
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logging.error(f"Invalid XML in sitemap {sitemap_url}: {e}")
            return set()

        # Handle different sitemap namespaces
        namespaces = {
            "sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "ns": "http://www.sitemaps.org/schemas/sitemap/0.9",
        }

        urls = set()

        # Look for sitemap index entries (nested sitemaps)
        sitemap_elements = root.findall(
            ".//sitemap:sitemap", namespaces
        ) + root.findall(".//ns:sitemap", namespaces)

        for sitemap_elem in sitemap_elements:
            if len(urls) >= config["max_urls"]:
                logging.info(f"Max URLs limit ({config['max_urls']}) reached")
                break

            loc_elem = sitemap_elem.find(
                "sitemap:loc", namespaces
            ) or sitemap_elem.find("ns:loc", namespaces)

            if loc_elem is not None and loc_elem.text:
                nested_sitemap_url = loc_elem.text.strip()

                # Validate URL is from same domain
                if not nested_sitemap_url.startswith(config["base_url"]):
                    logging.warning(f"Skipping external sitemap: {nested_sitemap_url}")
                    continue

                # Recursively parse nested sitemaps
                try:
                    nested_urls = parse_sitemap(
                        session, nested_sitemap_url, config, visited_sitemaps, depth + 1
                    )
                    urls.update(nested_urls)
                except Exception as e:
                    logging.error(
                        f"Error parsing nested sitemap {nested_sitemap_url}: {e}"
                    )
                    continue

        # Look for URL entries
        url_elements = root.findall(".//sitemap:url", namespaces) + root.findall(
            ".//ns:url", namespaces
        )

        for url_elem in url_elements:
            if len(urls) >= config["max_urls"]:
                logging.info(f"Max URLs limit ({config['max_urls']}) reached")
                break

            loc_elem = url_elem.find("sitemap:loc", namespaces) or url_elem.find(
                "ns:loc", namespaces
            )

            if loc_elem is not None and loc_elem.text:
                url = loc_elem.text.strip()

                # Validate URL and check for drinkware
                if (
                    url
                    and url.startswith(config["base_url"])
                    and "drinkware" in url.lower()
                ):
                    urls.add(url)
                    logging.info(f"Found drinkware URL: {url}")

        return urls

    except Exception as e:
        logging.error(f"Error parsing sitemap {sitemap_url}: {e}")
        return set()


def extract_prices_safely(card) -> List[str]:
    """Safely extract prices from product card"""
    prices = []

    try:
        # Primary selector: .product-card > .product-card__info > div > .price-list > sale-price
        sale_price_elements = card.select(
            ".product-card__info > div > .price-list > sale-price"
        )

        for sale_price in sale_price_elements:
            try:
                # Get all text content from the sale-price element
                price_text = sale_price.get_text(strip=True)
                logging.debug(f"Extracted sale-price text: {price_text}")

                # Extract numeric values more safely
                numeric_values = re.findall(r"\d+(?:\.\d{1,2})?", price_text)
                for value in numeric_values:
                    try:
                        float_val = float(value)
                        if 1.0 <= float_val <= 10000.0:  # Reasonable price range
                            prices.append(f"RM {value}")
                    except ValueError:
                        continue
            except Exception as e:
                logging.debug(f"Error processing sale-price element: {e}")
                continue

        # Fallback selectors if primary fails
        if not prices:
            fallback_selectors = [
                ".product-card__info > div > .price-list > sale-price > span",  # Original with span
                ".price-list span",
                ".price span",
                '[class*="price"] span',
                ".money",
            ]

            for selector in fallback_selectors:
                try:
                    elements = card.select(selector)
                    for elem in elements[:3]:  # Limit to avoid processing too many
                        elem_text = elem.get_text(strip=True)
                        numeric_values = re.findall(r"\d+(?:\.\d{1,2})?", elem_text)
                        for value in numeric_values:
                            try:
                                float_val = float(value)
                                if 1.0 <= float_val <= 10000.0:
                                    prices.append(f"RM {value}")
                                    if len(prices) >= 3:  # Limit prices per product
                                        break
                            except ValueError:
                                continue
                        if prices:
                            break
                    if prices:
                        break
                except Exception as e:
                    logging.debug(f"Error with fallback selector '{selector}': {e}")
                    continue

    except Exception as e:
        logging.warning(f"Error extracting prices: {e}")

    # Default price if none found
    if not prices:
        prices = ["RM 0.00"]

    return prices[:3]  # Limit to 3 prices max


def extract_product_from_card(card, base_url: str) -> Dict:
    """Extract product info from a product card element with validation"""
    try:
        # Extract product name and URL
        # Selector: .product-card > .product-card__info > div > .product-card__title > a
        title_link = card.select_one(
            ".product-card__info > div > .product-card__title > a"
        )

        if not title_link:
            logging.debug("No title link found in product card")
            return None

        # Get product name (text content of the link)
        product_name = title_link.get_text(strip=True)
        if not product_name or len(product_name) < 2:
            logging.debug("Invalid or missing product name")
            return None

        # Sanitize product name
        product_name = re.sub(r"[^\w\s\-\(\)\.&]", "", product_name)[
            :200
        ]  # Limit length

        # Get product URL (href attribute + base_url)
        href = title_link.get("href", "").strip()
        if not href:
            logging.debug("Missing href in title link")
            return None

        # Validate and construct URL
        try:
            product_url = urljoin(base_url, href)
            # Basic URL validation
            if not product_url.startswith(base_url):
                logging.warning(f"Invalid product URL: {product_url}")
                return None
        except Exception as e:
            logging.warning(f"Error constructing URL from href '{href}': {e}")
            return None

        # Extract prices with better validation
        prices = extract_prices_safely(card)

        product = {"name": product_name, "url": product_url, "price": prices}

        logging.debug(f"Extracted product: {product_name} - {prices}")
        return product

    except Exception as e:
        logging.warning(f"Error extracting product from card: {e}")
        return None


def scrape_product_page(
    session: requests.Session, url: str, config: dict
) -> List[Dict]:
    """Scrape products from a drinkware page with safety checks"""
    logging.info(f"Scraping products from: {url}")

    try:
        content = get_page_content(url, session, config)
        if not content:
            return []

        # Parse HTML safely
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:
            logging.error(f"HTML parsing error for {url}: {e}")
            return []

        products = []

        # Find all product cards using the specified CSS selector pattern
        try:
            product_cards = soup.select(".product-card")
        except Exception as e:
            logging.error(f"Error selecting product cards: {e}")
            return []

        logging.info(f"Found {len(product_cards)} product cards")

        # Limit number of products processed per page
        cards_to_process = product_cards[: config["max_products_per_page"]]

        for i, card in enumerate(cards_to_process):
            try:
                product = extract_product_from_card(card, config["base_url"])
                if product:
                    products.append(product)
                    logging.debug(f"Extracted product {i+1}: {product['name']}")
            except Exception as e:
                logging.warning(f"Error extracting product from card {i+1}: {e}")
                continue

        logging.info(f"Successfully extracted {len(products)} products from {url}")
        return products

    except Exception as e:
        logging.error(f"Error scraping product page {url}: {e}")
        return []


def validate_product(product: Dict, base_url: str) -> bool:
    """Validate product data"""
    try:
        # Check required fields
        if (
            not product.get("name")
            or not product.get("url")
            or not product.get("price")
        ):
            return False

        # Validate name
        name = product["name"]
        if not isinstance(name, str) or len(name.strip()) < 2:
            return False

        # Validate URL
        url = product["url"]
        if not isinstance(url, str) or not url.startswith(base_url):
            return False

        # Validate price
        prices = product["price"]
        if not isinstance(prices, list) or not prices:
            return False

        return True

    except Exception as e:
        logging.debug(f"Product validation error: {e}")
        return False


def deduplicate_products(products: List[Dict]) -> List[Dict]:
    """Remove duplicate products based on URL with logging"""
    seen_urls = set()
    unique_products = []

    for product in products:
        url = product.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_products.append(product)
        else:
            logging.debug(
                f"Duplicate product filtered: {product.get('name', 'Unknown')}"
            )

    logging.info(f"Deduplication: {len(products)} -> {len(unique_products)} products")
    return unique_products


def get_fallback_products(base_url: str) -> List[Dict]:
    """Comprehensive fallback products if scraping fails"""
    logging.info("Using fallback product data")
    return [
        {
            "name": "ZUS All Day Cup 500ml (17oz) - Aqua Collection",
            "url": f"{base_url}/products/zus-all-day-cup-500ml-17oz-aqua-collection",
            "price": ["RM 79.00"],
        },
        {
            "name": "ZUS All Day Cup 500ml (17oz) - Mountain Collection",
            "url": f"{base_url}/products/zus-all-day-cup-500ml-17oz-mountain-collection",
            "price": ["RM 79.00"],
        },
        {
            "name": "[Corak Malaysia] Tiga Sekawan Bundle",
            "url": f"{base_url}/products/corak-malaysia-tiga-sekawan-bundle",
            "price": ["RM 133.90"],
        },
        {
            "name": "[Kopi Patah Hati] ZUS Frozee Cold Cup 650ml (22oz)",
            "url": f"{base_url}/products/kopi-patah-hati-zus-frozee-cold-cup-650ml-22oz",
            "price": ["RM 44.00"],
        },
        {
            "name": "ZUS OG CUP 2.0 With Screw-On Lid 500ml (17oz)",
            "url": f"{base_url}/products/zus-og-cup-2-0-with-screw-on-lid",
            "price": ["RM 55.00"],
        },
        {
            "name": "ZUS All-Can Tumbler 600ml (20oz)",
            "url": f"{base_url}/products/zus-all-can-tumbler-600ml-20oz",
            "price": ["RM 105.00"],
        },
    ]


def scrape_products_main(config: dict) -> List[Dict]:
    """Main method to scrape all drinkware products with comprehensive safety"""
    logging.info(
        "Starting ZUS Coffee drinkware scraping using safe sitemap approach..."
    )

    try:
        session = create_session()
        sitemap_url = f"{config['base_url']}/sitemap.xml"
        visited_sitemaps = set()

        # Step 1: Parse sitemap to find drinkware URLs
        logging.info("Step 1: Parsing sitemap to find drinkware URLs...")

        start_time = time.time()
        drinkware_urls = parse_sitemap(session, sitemap_url, config, visited_sitemaps)
        parse_time = time.time() - start_time

        if not drinkware_urls:
            logging.warning("No drinkware URLs found in sitemap, using fallback")
            return get_fallback_products(config["base_url"])

        logging.info(f"Found {len(drinkware_urls)} drinkware URLs in {parse_time:.2f}s")

        # Limit URLs to process for safety
        urls_to_process = list(drinkware_urls)[: config["max_urls"]]
        if len(urls_to_process) < len(drinkware_urls):
            logging.info(f"Limited to {len(urls_to_process)} URLs for safety")

        # Step 2: Scrape products from each drinkware URL
        logging.info("Step 2: Scraping products from drinkware pages...")
        all_products = []

        for i, url in enumerate(urls_to_process, 1):
            try:
                logging.info(f"Processing URL {i}/{len(urls_to_process)}: {url}")

                products = scrape_product_page(session, url, config)
                if products:
                    all_products.extend(products)
                    logging.info(f"Got {len(products)} products from {url}")
                else:
                    logging.warning(f"No products found on {url}")

                # Safety: Check if we should continue
                if not check_request_limits(config["max_requests"]):
                    logging.warning("Request limit reached, stopping early")
                    break

                # Add delay between page requests
                if i < len(urls_to_process):  # Don't sleep after last request
                    delay = random.uniform(config["min_delay"], config["max_delay"])
                    logging.debug(f"Waiting {delay:.2f}s before next page...")
                    time.sleep(delay)

            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                continue

        # Step 3: Process and validate results
        if not all_products:
            logging.warning("No products found from any page, using fallback")
            return get_fallback_products(config["base_url"])

        unique_products = deduplicate_products(all_products)

        # Final validation
        valid_products = []
        for product in unique_products:
            if validate_product(product, config["base_url"]):
                valid_products.append(product)
            else:
                logging.debug(
                    f"Invalid product filtered out: {product.get('name', 'Unknown')}"
                )

        if not valid_products:
            logging.warning("No valid products after filtering, using fallback")
            return get_fallback_products(config["base_url"])

        logging.info(
            f"Successfully scraped {len(valid_products)} valid unique products"
        )
        logging.info(f"Total requests made: {_request_count}")

        return valid_products

    except Exception as e:
        logging.error(f"Error in main scraping process: {e}")
        return get_fallback_products(config["base_url"])


def scrape_zus_products() -> List[Dict]:
    """Main function to scrape ZUS products with comprehensive error handling"""
    try:
        global _request_count
        _request_count = 0  # Reset request counter

        logging.info("Attempting to scrape ZUS Coffee drinkware...")
        products = scrape_products_main(PRODUCT_CONFIG)

        if len(products) >= 3:  # If we got reasonable results
            logging.info(f"Successfully scraped {len(products)} products!")
            return products
        else:
            logging.warning(
                "Scraping returned insufficient products, using fallback data..."
            )
            return get_fallback_products(PRODUCT_CONFIG["base_url"])

    except Exception as e:
        # Fallback logging if scraper fails to initialize
        logging.error(f"Scraping failed with error: {e}")
        logging.info("Using fallback product data...")
        return get_fallback_products(PRODUCT_CONFIG["base_url"])


# Store scraping functions
def is_valid_store_pair(store_name: str, store_location: str) -> bool:
    """Validate that a store name and location pair is valid"""
    try:
        # Check if name or location is empty or too short
        if not store_name or not store_location:
            return False

        if len(store_name.strip()) < 2 or len(store_location.strip()) < 5:
            return False

        # Filter out entries where name or location is "Ingredients" or similar noise
        name_lower = store_name.lower().strip()
        location_lower = store_location.lower().strip()

        # Check for "Ingredients" or other unwanted entries
        unwanted_terms = ["ingredients", "ingredient", "allergen", "nutrition"]
        if any(term in name_lower for term in unwanted_terms) or any(
            term in location_lower for term in unwanted_terms
        ):
            return False

        # Only include stores where the location contains "Kuala Lumpur" or "Selangor"
        if not ("kuala lumpur" in location_lower or "selangor" in location_lower):
            return False

        return True

    except Exception as e:
        logging.debug(f"Store pair validation error: {e}")
        return False


def extract_store_info_from_elements(
    name_element, location_element, store_index: int
) -> Dict:
    """Extract store name and location from individual elements"""
    try:
        # Extract store name directly from the name element
        if not name_element:
            logging.debug(f"Store {store_index}: No name element found")
            return None

        store_name = name_element.get_text(strip=True)
        if not store_name or len(store_name) < 2:
            logging.debug(f"Store {store_index}: Invalid or missing store name")
            return None

        # Extract store location directly from the location element
        if not location_element:
            logging.debug(f"Store {store_index}: No location element found")
            return None

        store_location = location_element.get_text(strip=True)
        if not store_location or len(store_location) < 5:
            logging.debug(f"Store {store_index}: Invalid or missing store location")
            return None

        # Validate the pair again (double-check)
        if not is_valid_store_pair(store_name, store_location):
            logging.debug(f"Store {store_index}: Invalid store pair after extraction")
            return None

        # Sanitize store name
        store_name = re.sub(r"[^\w\s\-\(\)\.&,]", "", store_name)[:200]

        # Sanitize store location
        store_location = re.sub(r"[^\w\s\-\(\)\.&,#/]", "", store_location)[:500]

        # Format as requested: {name}, {location}
        formatted_store = f"{store_name}, {store_location}"

        store = {
            "name": store_name,
            "location": store_location,
            "formatted": formatted_store,
        }

        logging.debug(f"Extracted store {store_index}: {formatted_store}")
        return store

    except Exception as e:
        logging.warning(f"Error extracting store info for store {store_index}: {e}")
        return None


def scrape_stores_from_page(
    session: requests.Session, page_num: int, config: dict
) -> List[Dict]:
    """Scrape store locations from a specific page"""
    url = f"{config['base_url']}/page/{page_num}/"
    logging.info(f"Scraping stores from page {page_num}: {url}")

    try:
        content = get_page_content(url, session, config)
        if not content:
            return []

        # Parse HTML safely
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:
            logging.error(f"HTML parsing error for {url}: {e}")
            return []

        stores = []

        # Find the main container
        try:
            main_container = soup.select_one(".category-kuala-lumpur-selangor")
            if not main_container:
                logging.warning(f"Main container not found on page {page_num}")
                return []

            # Find all store containers (each store has its own container)
            # Look for containers with both name and location sections
            store_containers = main_container.select("div")
            logging.info(
                f"Found {len(store_containers)} potential store containers on page {page_num}"
            )

            processed_stores = set()  # Track processed stores to avoid duplicates

            for container_index, container in enumerate(store_containers):
                try:
                    # Extract name from: section (first child) > div > div > div > div (first child) > div > p
                    name_element = container.select_one(
                        "section:first-child div div div div:first-child div p"
                    )

                    # Extract location from: section (second child) > div > div > div > div > div > p
                    location_element = container.select_one(
                        "section:nth-child(2) div div div div div p"
                    )

                    if name_element and location_element:
                        store_name = name_element.get_text(strip=True)
                        store_location = location_element.get_text(strip=True)

                        # Filter out unwanted entries and validate pairing
                        if is_valid_store_pair(store_name, store_location):
                            # Create a unique key to avoid duplicates
                            store_key = f"{store_name.strip()}-{store_location.strip()}"

                            if store_key not in processed_stores:
                                store = extract_store_info_from_elements(
                                    name_element,
                                    location_element,
                                    len(processed_stores) + 1,
                                )
                                if store:
                                    stores.append(store)
                                    processed_stores.add(store_key)
                                    logging.debug(
                                        f"Extracted store: {store['name']} - {store['location']}"
                                    )
                            else:
                                logging.debug(f"Duplicate store skipped: {store_name}")
                        else:
                            logging.debug(
                                f"Filtered out invalid store pair: '{store_name}' / '{store_location}'"
                            )

                except Exception as e:
                    logging.warning(
                        f"Error extracting store info from container {container_index + 1}: {e}"
                    )
                    continue

        except Exception as e:
            logging.error(f"Error finding store sections on page {page_num}: {e}")
            return []

        logging.info(
            f"Successfully extracted {len(stores)} stores from page {page_num}"
        )
        return stores

    except Exception as e:
        logging.error(f"Error scraping page {page_num}: {e}")
        return []


def validate_store(store: Dict) -> bool:
    """Validate store data"""
    try:
        # Check required fields
        if (
            not store.get("name")
            or not store.get("location")
            or not store.get("formatted")
        ):
            return False

        # Validate name
        name = store["name"]
        if not isinstance(name, str) or len(name.strip()) < 2:
            return False

        # Validate location
        location = store["location"]
        if not isinstance(location, str) or len(location.strip()) < 5:
            return False

        # Validate formatted
        formatted = store["formatted"]
        if not isinstance(formatted, str) or "," not in formatted:
            return False

        return True

    except Exception as e:
        logging.debug(f"Store validation error: {e}")
        return False


def deduplicate_stores(stores: List[Dict]) -> List[Dict]:
    """Remove duplicate stores based on formatted string"""
    seen_formatted = set()
    unique_stores = []

    for store in stores:
        formatted = store.get("formatted", "")
        if formatted and formatted not in seen_formatted:
            seen_formatted.add(formatted)
            unique_stores.append(store)
        else:
            logging.debug(f"Duplicate store filtered: {store.get('name', 'Unknown')}")

    logging.info(f"Store deduplication: {len(stores)} -> {len(unique_stores)} stores")
    return unique_stores


def get_fallback_stores() -> List[Dict]:
    """Fallback store data if scraping fails"""
    logging.info("Using fallback store data")
    return [
        {
            "name": "ZUS Coffee – Temu Business Centre City Of Elmina",
            "location": "No 5 (Ground Floor), Jalan Eserina AA U16/AA Elmina, East, Seksyen U16, 40150 Shah Alam, Selangor",
            "formatted": "ZUS Coffee – Temu Business Centre City Of Elmina, No 5 (Ground Floor), Jalan Eserina AA U16/AA Elmina, East, Seksyen U16, 40150 Shah Alam, Selangor",
        },
        {
            "name": "ZUS Coffee – Spectrum Shopping Mall",
            "location": "Lot CW-5 Cafe Walk, Ground Floor Spectrum Shopping Mall Jalan Wawasan Ampang, 4, 2, Bandar Baru Ampang, 68000 Ampang, Selangor",
            "formatted": "ZUS Coffee – Spectrum Shopping Mall, Lot CW-5 Cafe Walk, Ground Floor Spectrum Shopping Mall Jalan Wawasan Ampang, 4, 2, Bandar Baru Ampang, 68000 Ampang, Selangor",
        },
        {
            "name": "ZUS Coffee – Bandar Menjalara",
            "location": "37, Jalan 3/62a, Bandar Menjalara, 52200 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
            "formatted": "ZUS Coffee – Bandar Menjalara, 37, Jalan 3/62a, Bandar Menjalara, 52200 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
        },
        {
            "name": "ZUS Coffee – Jabatan Peguam Negara, Putrajaya",
            "location": "Bangunan Jabatan Peguam Negara AGC Persint 4, Lot 1, Level 1, Putrajaya 62100 Malaysia",
            "formatted": "ZUS Coffee – Jabatan Peguam Negara, Putrajaya, Bangunan Jabatan Peguam Negara AGC Persint 4, Lot 1, Level 1, Putrajaya 62100 Malaysia",
        },
        {
            "name": "ZUS Coffee – LSH33, Sentul",
            "location": "G-11, Ground Floor, Laman Seri Harmoni (LSH33), No. 3, Jalan Batu Muda Tambahan 3, Sentul, 51100 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
            "formatted": "ZUS Coffee – LSH33, Sentul, G-11, Ground Floor, Laman Seri Harmoni (LSH33), No. 3, Jalan Batu Muda Tambahan 3, Sentul, 51100 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",
        },
    ]


def scrape_all_stores(config: dict) -> List[Dict]:
    """Main method to scrape all store locations from paginated pages"""
    logging.info(
        f"Starting ZUS Coffee store location scraping from pages {config['min_page']} to {config['max_pages']}..."
    )

    try:
        session = create_session()
        all_stores = []

        for page_num in range(config["min_page"], config["max_pages"] + 1):
            try:
                logging.info(f"Processing page {page_num}/{config['max_pages']}")

                stores = scrape_stores_from_page(session, page_num, config)
                if stores:
                    all_stores.extend(stores)
                    logging.info(f"Got {len(stores)} stores from page {page_num}")
                else:
                    logging.warning(f"No stores found on page {page_num}")

                # Safety: Check if we should continue
                if not check_request_limits(config["max_requests"]):
                    logging.warning("Request limit reached, stopping early")
                    break

                # Add delay between page requests
                if page_num < config["max_pages"]:  # Don't sleep after last request
                    delay = random.uniform(config["min_delay"], config["max_delay"])
                    logging.debug(f"Waiting {delay:.2f}s before next page...")
                    time.sleep(delay)

            except Exception as e:
                logging.error(f"Error scraping page {page_num}: {e}")
                continue

        # Process and validate results
        if not all_stores:
            logging.warning("No stores found from any page, using fallback")
            return get_fallback_stores()

        unique_stores = deduplicate_stores(all_stores)

        # Final validation
        valid_stores = []
        for store in unique_stores:
            if validate_store(store):
                valid_stores.append(store)
            else:
                logging.debug(
                    f"Invalid store filtered out: {store.get('name', 'Unknown')}"
                )

        if not valid_stores:
            logging.warning("No valid stores after filtering, using fallback")
            return get_fallback_stores()

        logging.info(f"Successfully scraped {len(valid_stores)} valid unique stores")
        logging.info(f"Total requests made: {_request_count}")

        return valid_stores

    except Exception as e:
        logging.error(f"Error in main store scraping process: {e}")
        return get_fallback_stores()


def scrape_zus_stores() -> List[Dict]:
    """Main function to scrape ZUS store locations with comprehensive error handling"""
    try:
        global _request_count
        _request_count = 0  # Reset request counter

        logging.info("Attempting to scrape ZUS Coffee store locations...")
        stores = scrape_all_stores(STORE_CONFIG)

        if len(stores) >= 3:  # If we got reasonable results
            logging.info(f"Successfully scraped {len(stores)} stores!")
            return stores
        else:
            logging.warning(
                "Scraping returned insufficient stores, using fallback data..."
            )
            return get_fallback_stores()

    except Exception as e:
        # Fallback logging if scraper fails to initialize
        logging.error(f"Store scraping failed with error: {e}")
        logging.info("Using fallback store data...")
        return get_fallback_stores()


if __name__ == "__main__":
    # Test both scrapers with proper logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Test product scraper
        print("=== Testing Product Scraper ===")
        products = scrape_zus_products()

        print(f"\nFound {len(products)} products:")
        for i, product in enumerate(products[:5], 1):
            prices_str = (
                ", ".join(product["price"])
                if isinstance(product["price"], list)
                else product["price"]
            )
            print(f"{i}. {product['name']} - {prices_str}")
            print(f"   URL: {product['url']}")

        if len(products) > 5:
            print(f"... and {len(products) - 5} more products")

        print("\n" + "=" * 50)

        # Test store scraper
        print("=== Testing Store Scraper ===")
        stores = scrape_zus_stores()

        print(f"\nFound {len(stores)} store locations:")
        for i, store in enumerate(stores[:5], 1):
            print(f"{i}. {store['formatted']}")

        if len(stores) > 5:
            print(f"... and {len(stores) - 5} more store locations")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.error(f"Unexpected error in main: {e}")
