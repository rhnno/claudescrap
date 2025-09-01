"""
Enhanced ML-Powered Web Scraper Package

This package contains all the modules needed for intelligent web scraping with 
ML-powered pagination detection and configurable analysis.
"""

# Core browser and scraping functionality
from .browser import BrowserManager
from .storage import DataStorage
from .utils import RandomUtils

# Enhanced ML-powered analyzer components
from .analyzer import ConfigurableAnalyzer, EnhancedConfigurableAnalyzer
from .analyzer import TrainingDataCollector

# Scraper components
from .scraper import SmartTokopediaScraper, EnhancedTokopediaScraper

# Legacy components (if they exist)
try:
    from .parser import ProductParser
except ImportError:
    ProductParser = None

# Test module (optional)
try:
    from .test import *
except ImportError:
    pass

# Package metadata
__version__ = "2.0.0"
__author__ = "Enhanced Analyzer System"
__description__ = "ML-powered web scraper with configurable pagination detection"

# What gets imported when someone does: from func import *
__all__ = [
    # Core components
    'BrowserManager',
    'DataStorage', 
    'RandomUtils',
    
    # Enhanced ML components
    'ConfigurableAnalyzer',
    'EnhancedConfigurableAnalyzer',
    'TrainingDataCollector',
    
    # Scraper components
    'SmartTokopediaScraper',
    'EnhancedTokopediaScraper',

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

