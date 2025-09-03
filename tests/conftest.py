"""
Test configuration and fixtures for claudescrap tests.

This module provides common test fixtures, mocks, and configuration
for testing the ScraperService and related components.
"""
import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any
import tempfile
import os
import sys

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.scraper_service import ScraperService
from src.models.database import DatabaseManager, ScrapingJob
from src.ace import ScrapingOrchestrator


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_database_manager():
    """Mock DatabaseManager for testing."""
    mock_db = Mock(spec=DatabaseManager)
    mock_db.create_job = Mock(return_value=Mock())
    mock_db.update_job_status = Mock()
    mock_db.save_products = Mock()
    mock_db.get_session = Mock()
    
    # Mock session and query chain
    mock_session = Mock()
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = []
    mock_query.first.return_value = None
    mock_session.close = Mock()
    mock_db.get_session.return_value = mock_session
    
    return mock_db


@pytest.fixture
def mock_scraping_orchestrator():
    """Mock ScrapingOrchestrator for testing."""
    mock_orchestrator = Mock(spec=ScrapingOrchestrator)
    mock_orchestrator.setup_browser = Mock(return_value=True)
    mock_orchestrator.close = Mock()
    mock_orchestrator._scrape_query = Mock(return_value=[
        {
            'name': 'Test Product 1',
            'price': '$100',
            'url': 'https://test.com/product1'
        },
        {
            'name': 'Test Product 2', 
            'price': '$200',
            'url': 'https://test.com/product2'
        }
    ])
    mock_orchestrator.session_stats = {'total_pages': 2}
    return mock_orchestrator


@pytest.fixture
def mock_scraping_job():
    """Mock ScrapingJob for testing."""
    job = Mock(spec=ScrapingJob)
    job.job_id = 'test-job-123'
    job.status = 'pending'
    job.site = 'tokopedia'
    job.query = 'test laptop'
    job.current_page = 0
    job.total_pages = 5
    job.products_found = 0
    job.created_at = datetime.utcnow()
    job.completed_at = None
    job.error_message = None
    return job


@pytest.fixture
def sample_products():
    """Sample product data for testing."""
    return [
        {
            'name': 'Gaming Laptop',
            'price': '$1299',
            'url': 'https://tokopedia.com/product1',
            'job_id': 'test-job-123',
            'site': 'tokopedia',
            'query': 'gaming laptop'
        },
        {
            'name': 'Business Laptop',
            'price': '$899',
            'url': 'https://tokopedia.com/product2',
            'job_id': 'test-job-123',
            'site': 'tokopedia',
            'query': 'gaming laptop'
        }
    ]


@pytest.fixture
def scraper_service_with_mocks(mock_database_manager):
    """ScraperService instance with mocked dependencies."""
    with patch('src.services.scraper_service.DatabaseManager', return_value=mock_database_manager):
        with patch('src.services.scraper_service.Utils') as mock_utils:
            service = ScraperService()
            service.db = mock_database_manager
            return service


@pytest.fixture
def temp_test_db():
    """Temporary database for integration testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    # Set up test database
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    if 'DATABASE_URL' in os.environ:
        del os.environ['DATABASE_URL']


class AsyncContextManager:
    """Helper for testing async context managers."""
    
    def __init__(self, mock_obj):
        self.mock_obj = mock_obj
    
    async def __aenter__(self):
        return self.mock_obj
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockBrowserPool:
    """Mock browser pool for testing."""
    
    def __init__(self):
        self.browsers = []
        self.max_size = 2
        self.lock_acquired = False
    
    async def __aenter__(self):
        self.lock_acquired = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.lock_acquired = False
    
    def pop(self):
        if self.browsers:
            return self.browsers.pop()
        raise IndexError("No browsers available")
    
    def append(self, browser):
        if len(self.browsers) < self.max_size:
            self.browsers.append(browser)


# Test data constants
TEST_JOB_ID = 'test-job-12345'
TEST_SITE = 'tokopedia'
TEST_QUERY = 'gaming laptop'
TEST_MAX_PAGES = 5

TEST_SESSION_STATS = {
    'total_jobs': 10,
    'successful_jobs': 8,
    'failed_jobs': 2,
    'running_jobs': 1,
    'total_products': 156,
    'total_pages': 25,
    'success_rate': 80.0
}