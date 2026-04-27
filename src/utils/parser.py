"""Product Parser Module for HTML Content Extraction.

This module provides the ProductParser class for extracting structured product data
from HTML content using BeautifulSoup. It includes configurable CSS selectors
for different e-commerce sites and robust error handling.

Example:
    Basic usage of the product parser::

        parser = ProductParser()
        products = parser.parse_single_file('tokopedia_page.html')
        all_products = parser.parse_all_files('data/raw_html')

Note:
    The parser is optimized for Tokopedia's HTML structure but can be
    configured for other e-commerce sites by updating the selectors.
"""
from bs4 import BeautifulSoup, Tag
import html
import os
from typing import List, Dict, Any, Optional, Union

class ProductParser:
    """Parses HTML content to extract structured product data.
    
    This class provides methods to extract product information from HTML content
    using configurable CSS selectors. It supports parsing single files or
    batch processing of multiple HTML files.
    
    Attributes:
        selectors (Dict[str, str]): CSS selectors for different product elements
    
    Example:
        >>> parser = ProductParser()
        >>> products = parser.parse_single_file('product_page.html')
        >>> print(f"Found {len(products)} products")
    
    Note:
        Default selectors are configured for Tokopedia's HTML structure.
        Selectors can be modified for other e-commerce sites.
    """
    
    def __init__(self) -> None:
        """Initialize ProductParser with default CSS selectors.
        
        Sets up CSS selectors for extracting various product elements
        from Tokopedia's HTML structure.
        
        Note:
            Selectors can be modified after initialization to support
            other e-commerce sites with different HTML structures.
        """
        self.selectors = {
            'product_cards': 'div.css-5wh65g',
            'title': 'div.css-1f4mp12',
            'price': 'div.css-rhd610', 
            'discount': 'span._7UCYdN8MrOTwg0MKcGu8zg==',
            'link': 'a[href]',
            'location': 'span.gxi+fsEljOjqhjSKqjE+sw==.flip',
            'shop': 'span.si3CNdiG8AR0EaXvf6bFbQ==',
            'sold': 'span.u6SfjDD2WiBlNW7zHmzRhQ==',
            'rating': 'span._2NfJxPu4JC-55aCJ8bEsyw==',
            'original_price': 'span.hC1B8wTAoPszbEZj80w6Qw=='
        }
    
    def parse_single_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse a single HTML file and extract product data.
        
        Reads an HTML file, parses it with BeautifulSoup, and extracts
        product information using configured CSS selectors.
        
        Args:
            filepath (str): Path to the HTML file to parse
        
        Returns:
            List[Dict[str, Any]]: List of product dictionaries with extracted data
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            UnicodeDecodeError: If file encoding is not UTF-8
            Exception: Other file reading or parsing errors
        
        Example:
            >>> parser = ProductParser()
            >>> products = parser.parse_single_file('tokopedia_search.html')
            >>> for product in products:
            ...     print(f"{product['Product Title']}: {product['Price']}")
        
        Note:
            Files are expected to be in UTF-8 encoding. HTML content
            is parsed using BeautifulSoup's html.parser.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            return self._extract_products(soup)
        
        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
            return []
    
    def parse_all_files(self, folder: str = "data/raw_html") -> List[Dict[str, Any]]:
        """Parse all HTML files in a folder and extract product data.
        
        Processes all .html files in the specified folder and aggregates
        product data from all files into a single list.
        
        Args:
            folder (str, optional): Path to folder containing HTML files.
                Defaults to "data/raw_html".
        
        Returns:
            List[Dict[str, Any]]: Combined list of all products from all files
        
        Example:
            >>> parser = ProductParser()
            >>> all_products = parser.parse_all_files('scraped_pages')
            >>> print(f"Total products from all files: {len(all_products)}")
        
        Note:
            Only processes files with .html extension. Provides progress
            logging for each file processed.
        """
        all_listings: List[Dict[str, Any]] = []
        
        if not os.path.exists(folder):
            print(f"❌ Folder {folder} doesn't exist")
            return all_listings
        
        html_files = [f for f in os.listdir(folder) if f.endswith('.html')]
        
        for filename in html_files:
            filepath = os.path.join(folder, filename)
            print(f"📝 Parsing: {filename}")
            
            listings = self.parse_single_file(filepath)
            all_listings.extend(listings)
        
        print(f"✅ Total products found: {len(all_listings)}")
        return all_listings
    
    def _extract_products(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract product data from BeautifulSoup object.
        
        Finds all product cards in the parsed HTML and extracts
        product information from each card.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
        
        Returns:
            List[Dict[str, Any]]: List of extracted product data
        
        Example:
            >>> soup = BeautifulSoup(html_content, 'html.parser')
            >>> products = parser._extract_products(soup)
        
        Note:
            Uses the 'product_cards' selector to find product containers.
            Continues processing even if individual product extraction fails.
        """
        listings = []
        product_cards = soup.find_all('div', class_='css-5wh65g')
        
        for card in product_cards:
            try:
                product_data = self._extract_single_product(card)
                if product_data:
                    listings.append(product_data)
            except Exception as e:
                print(f"❌ Error extracting product: {e}")
                continue
        
        return listings
    
    def _extract_single_product(self, card: Tag) -> Optional[Dict[str, Any]]:
        """Extract data from a single product card.
        
        Extracts all available product information from a single product
        card using the configured CSS selectors.
        
        Args:
            card (Tag): BeautifulSoup element representing a product card
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary with product data or None if extraction fails
        
        Example:
            >>> card = soup.find('div', class_='css-5wh65g')
            >>> product = parser._extract_single_product(card)
            >>> print(product['Product Title'])
        
        Note:
            Uses safe_extract helper function to handle missing elements gracefully.
            Returns 'N/A' for missing or failed extractions.
        """
        def safe_extract(selector_key: str, attr: str = 'text') -> str:
            try:
                element = card.find(self.selectors[selector_key].split('.')[0], 
                                  class_='.'.join(self.selectors[selector_key].split('.')[1:]))
                if element and isinstance(element, Tag):
                    if attr == 'href':
                        href_value = element.get('href')
                        return str(href_value) if href_value else 'N/A'
                    else:
                        return html.unescape(element.get_text(strip=True))
                return 'N/A'
            except Exception:
                return 'N/A'
        
        return {
            'Product Title': safe_extract('title'),
            'Price': safe_extract('price'),
            'Sold': safe_extract('sold'),
            'discount': safe_extract('discount'),
            'Before Discount Price': safe_extract('original_price'),
            'Shop Name': safe_extract('shop'),
            'location': safe_extract('location'),
            'Rating': safe_extract('rating'),
            'Link Product': safe_extract('link', 'href')
        }