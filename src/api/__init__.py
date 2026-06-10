# API package
"""
API package initialization
"""

try:
    from .scraping_api import app, security, S
except ImportError :
    app = None


# Package metadata
__version__ = "1.0.0"
__author__ = "rhnno"
__description__ = "Scraping API server"

__all__ = [
    'app',
    'router'
]
