"""Data Storage Module for Scraping Operations.

This module provides the DataStorage class for handling file operations
related to web scraping, including saving HTML content and processed
data to various formats.

Example:
    Basic usage of the data storage::

        storage = DataStorage()
        storage.save_html(html_content, "laptop", 1)
        storage.save_to_csv(product_list, "gaming_laptop")

Note:
    Automatically creates necessary directory structure and handles
    file encoding for proper Unicode support.
"""
import os
import csv
from typing import List, Dict, Any

class DataStorage:
    """Handles all file operations for scraping data.
    
    This class manages the storage of HTML content and processed data
    from web scraping operations. It provides methods for saving raw
    HTML and converting processed data to CSV format.
    
    Attributes:
        html_folder (str): Directory path for storing raw HTML files
        csv_folder (str): Directory path for storing processed CSV files
    
    Example:
        >>> storage = DataStorage()
        >>> storage.save_html("<html>...</html>", "laptop", 1)
        >>> storage.save_to_csv(products, "gaming_laptop")
    
    Note:
        Automatically creates necessary directories if they don't exist.
        All files are saved with UTF-8 encoding for Unicode support.
    """
    
    def __init__(self) -> None:
        """Initialize DataStorage with default folder structure.
        
        Sets up the directory paths for HTML and CSV storage and
        ensures the directories exist.
        
        Note:
            Creates 'data/raw_html' and 'data/processed' directories
            if they don't already exist.
        """
        self.html_folder = "data/raw_html"
        self.csv_folder = "data/processed"
        self._ensure_folders()
    
    def _ensure_folders(self) -> None:
        """Create necessary folders if they don't exist.
        
        Creates the HTML and CSV storage directories using os.makedirs
        with exist_ok=True to avoid errors if directories already exist.
        
        Note:
            Called automatically during initialization to ensure
            proper directory structure is available.
        """
        os.makedirs(self.html_folder, exist_ok=True)
        os.makedirs(self.csv_folder, exist_ok=True)
    
    def save_html(self, html_content: str, query: str, page_num: int) -> str:
        """Save HTML content to file with structured naming.
        
        Saves raw HTML content to a file with a standardized naming
        convention for easy organization and retrieval.
        
        Args:
            html_content (str): Raw HTML content to save
            query (str): Search query used for this content
            page_num (int): Page number for this content
        
        Returns:
            str: Full path to the saved HTML file
        
        Raises:
            IOError: If file writing fails
            UnicodeEncodeError: If content encoding fails
        
        Example:
            >>> storage = DataStorage()
            >>> path = storage.save_html("<html>...</html>", "laptop", 1)
            >>> print(f"Saved to: {path}")
        
        Note:
            Files are named with pattern: page_{query}_{page_num:03d}.html
            All content is saved with UTF-8 encoding.
        """
        filename = f"page_{query}_{page_num:03d}.html"
        filepath = os.path.join(self.html_folder, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def save_to_csv(self, listings: List[Dict[str, Any]], query: str) -> str:
        """Save product listings to CSV file with predefined structure.
        
        Converts a list of product dictionaries to a CSV file with
        standardized column headers and UTF-8 encoding.
        
        Args:
            listings (List[Dict[str, Any]]): List of product dictionaries to save
            query (str): Search query used for this data (used in filename)
        
        Returns:
            str: Full path to the saved CSV file
        
        Raises:
            IOError: If file writing fails
            KeyError: If required product fields are missing
        
        Example:
            >>> products = [
            ...     {'Product Title': 'Gaming Laptop', 'Price': 'Rp 15.000.000'},
            ...     {'Product Title': 'Office Laptop', 'Price': 'Rp 8.000.000'}
            ... ]
            >>> storage = DataStorage()
            >>> path = storage.save_to_csv(products, "laptop")
        
        Note:
            CSV includes standard e-commerce fields: title, price, sold count,
            discount, original price, shop name, location, rating, and product link.
        """
        filename = f"Product_Data_{query}.csv"
        filepath = os.path.join(self.csv_folder, filename)
        
        fieldnames = [
            'Product Title', 'Price', 'Sold', 'discount', 
            'Before Discount Price', 'Shop Name', 'location', 
            'Rating', 'Link Product'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(listings)
        
        print(f"✅ CSV saved: {filepath}")
        return filepath

