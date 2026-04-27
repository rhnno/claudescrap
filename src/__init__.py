"""
Enhanced ML-Powered Web Scraper Package

This package contains all the modules needed for intelligent web scraping with 
ML-powered pagination detection and configurable analysis.
"""

# Core browser and scraping functionality
from .utils.browser import BrowserManager
from .utils.storage import DataStorage
from .utils.utils import RandomUtils

# Enhanced ML-powered analyzer components
# The ML-powered analyzer components have been removed as per the user's request.

# Scraper components

# Legacy components (if they exist)
try:
    from .utils.parser import ProductParser
except ImportError:
    ProductParser = None

# Package metadata
__version__ = "2.0.0"
__author__ = "rhnno"
__description__ = "web scraper app"

# What gets imported when someone does: from func import *
__all__ = [
    # Core components
    'BrowserManager',
    'DataStorage', 
    'RandomUtils',
    

    # Legacy components (if available)
    'ProductParser',
]

# Remove None values from __all__ (for missing legacy components)
__all__ = [item for item in __all__ if globals().get(item) is not None]

# Package initialization (simplified to avoid hanging)
def _initialize_package():
    """Initialize the enhanced scraper package"""
    import os
    
    # Create necessary directories silently
    directories = [
        'data',
        'data/raw_html',
        'data/processed', 
        'logs',
        'models',
        'config'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except:
            pass  # Ignore errors during directory creation

# Initialize package silently
try:
    _initialize_package()
except:
    pass  # Ignore initialization errors

# Cleanup
try:
    del _initialize_package
except:
    pass

