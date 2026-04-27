"""Models package for the claudescrap project.

This package contains database models and managers for storing
and retrieving scraping job data and results.
"""

from .database import DatabaseManager, ScrapingJob, Product, Base

__all__ = [
    'DatabaseManager',
    'ScrapingJob', 
    'Product',
    'Base',
]

__version__ = "1.0.0"
__author__ = "Claudescrap Project"