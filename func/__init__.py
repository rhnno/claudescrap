"""
Tokopedia Scraper Package

This package contains all the modules needed for scraping Tokopedia product data.
"""

# Import classes and functions you want to expose at package level
from .browser import BrowserManager
from .scraper import TokopediaScraper
from .parser import ProductParser
from .storage import DataStorage
from .utils import RandomUtils
from .analyzer import PaginationAnalyzer
from .analyzer import SmartTokopediaScraper
from .analyzer import TrainingDataCollector

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"

# What gets imported when someone does: from func import *
__all__ = [
    'BrowserManager',
    'TokopediaScraper', 
    'ProductParser',
    'DataStorage',
    'RandomUtils'
]

# Package initialization code (runs once when package is first imported)
print("🚀 Tokopedia Scraper Package Loaded")

