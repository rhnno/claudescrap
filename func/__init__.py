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
from .analyzer import ConfigurableAnalyzer
from .analyzer import SmartTokopediaScraper
from .analyzer import TrainingDataCollector

# Legacy components (if they exist)
try:
    from .scraper import TokopediaScraper
except ImportError:
    TokopediaScraper = None

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
    'SmartTokopediaScraper',
    'TrainingDataCollector',
    
    # Legacy components (if available)
    'TokopediaScraper',
    'ProductParser',
]

# Remove None values from __all__ (for missing legacy components)
__all__ = [item for item in __all__ if globals().get(item) is not None]

# Package initialization
def _initialize_package():
    """Initialize the enhanced scraper package"""
    import os
    
    # Create necessary directories
    directories = [
        'data',
        'data/raw_html',
        'data/processed', 
        'logs',
        'models',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Print initialization message
    print("🧠 Enhanced ML-Powered Web Scraper Package Loaded")
    print(f"📦 Version: {__version__}")
    print(f"🔧 Available components: {len(__all__)}")
    
    # Show available components
    components = {
        'Core': ['BrowserManager', 'DataStorage', 'RandomUtils'],
        'ML-Powered': ['ConfigurableAnalyzer', 'SmartTokopediaScraper', 'TrainingDataCollector'],
        'Legacy': ['TokopediaScraper', 'ProductParser']
    }
    
    for category, items in components.items():
        available = [item for item in items if item in __all__]
        if available:
            print(f"   {category}: {', '.join(available)}")

# Initialize package
_initialize_package()

# Cleanup
del _initialize_package

