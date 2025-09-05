"""
Test configuration and fixtures for claudescrap tests.

This module provides common test fixtures, mocks, and configuration
for testing the ScraperService and related components.
"""
import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, List, Any
import tempfile
import os
import sys

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.scraper_service import ScraperService
from src.models.database import DatabaseManager, ScrapingJob, Base
from src.ace import ScrapingOrchestrator


# PostgreSQL testing with testing.postgresql
try:
    import testing.postgresql
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    POSTGRES_TESTING_AVAILABLE = True
except ImportError:
    POSTGRES_TESTING_AVAILABLE = False


@pytest.fixture(scope="session")
def postgres_server():
    """Create a temporary PostgreSQL server for testing."""
    if not POSTGRES_TESTING_AVAILABLE:
        pytest.skip("testing.postgresql not available")
    
    try:
        # Create temporary PostgreSQL server
        postgres = testing.postgresql.Postgresql()
        yield postgres
        # Cleanup happens automatically when the object is destroyed
    except RuntimeError as e:
        if "command not found" in str(e):
            pytest.skip(f"PostgreSQL binaries not available: {e}")
        else:
            raise


@pytest.fixture(scope="session")  
def postgres_database_url(postgres_server):
    """Get database URL from temporary PostgreSQL server."""
    return postgres_server.url()


@pytest.fixture
def postgres_database_manager(postgres_database_url):
    """DatabaseManager connected to temporary PostgreSQL."""
    if not POSTGRES_TESTING_AVAILABLE:
        pytest.skip("testing.postgresql not available")
    
    # Create DatabaseManager with temporary database
    db_manager = DatabaseManager(database_url=postgres_database_url)
    
    yield db_manager
    
    # Cleanup - drop all tables
    try:
        Base.metadata.drop_all(bind=db_manager.engine)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def temp_test_db():
    """Intelligent test database fixture that adapts to environment.
    
    - In CI environments with DATABASE_URL: uses CI database
    - In environments with PostgreSQL binaries: uses testing.postgresql  
    - Otherwise: uses mock database for fast local testing
    
    This provides test isolation while being environment-aware.
    """
    # Check if we're in CI with DATABASE_URL set
    if os.getenv('DATABASE_URL') and os.getenv('CI'):
        try:
            db_manager = DatabaseManager(database_url=os.getenv('DATABASE_URL'))
            yield db_manager
            # Cleanup in CI
            try:
                Base.metadata.drop_all(bind=db_manager.engine)
            except Exception:
                pass
            return
        except Exception:
            pass  # Fall through to alternatives
    
    # Try testing.postgresql if available and PostgreSQL binaries exist
    if POSTGRES_TESTING_AVAILABLE:
        try:
            postgres = testing.postgresql.Postgresql()
            db_manager = DatabaseManager(database_url=postgres.url())
            yield db_manager
            # Cleanup happens automatically
            return
        except RuntimeError as e:
            if "command not found" in str(e):
                pass  # Fall through to mock
            else:
                raise
    
    # Fall back to mock database for local development
    mock_db = Mock(spec=DatabaseManager)
    mock_job = Mock(spec=ScrapingJob)
    mock_job.job_id = 'test-job-123'
    mock_job.status = 'completed'
    mock_job.current_page = 1
    mock_job.total_pages = 1
    mock_job.products_found = 1
    mock_job.created_at = datetime.now()
    mock_job.error_message = None
    
    mock_db.create_job = Mock(return_value=mock_job)
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
    mock_query.all.return_value = [mock_job]
    mock_query.first.return_value = mock_job  # Return the mock job
    mock_session.close = Mock()
    mock_db.get_session.return_value = mock_session
    
    yield mock_db


@pytest.fixture
def ci_database_manager():
    """Database manager for CI environment - uses DATABASE_URL."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        pytest.skip("DATABASE_URL not set in CI environment")
    
    # Create DatabaseManager with CI database
    db_manager = DatabaseManager(database_url=database_url)
    
    yield db_manager
    
    # Cleanup - drop all tables in CI
    try:
        Base.metadata.drop_all(bind=db_manager.engine)
    except Exception:
        pass  # Ignore cleanup errors in CI


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
    mock_job = Mock(spec=ScrapingJob)
    mock_job.job_id = 'test-job-123'
    mock_job.status = 'completed'
    mock_job.current_page = 1
    mock_job.total_pages = 1
    mock_job.products_found = 1
    mock_job.created_at = datetime.now()
    mock_job.error_message = None
    
    mock_db.create_job = Mock(return_value=mock_job)
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
    mock_query.all.return_value = [mock_job]
    mock_query.first.return_value = mock_job
    mock_session.close = Mock()
    mock_db.get_session.return_value = mock_session
    
    return mock_db


@pytest.fixture
def mock_scraping_orchestrator():
    """Mock ScrapingOrchestrator for testing."""
    mock_orchestrator = Mock(spec=ScrapingOrchestrator)
    mock_orchestrator.setup_browser = AsyncMock(return_value=True)
    mock_orchestrator.close = Mock()  # Changed from AsyncMock to Mock
    mock_orchestrator._scrape_query = Mock(return_value=[  # Changed from AsyncMock to Mock
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
    job.created_at = datetime.now(timezone.utc)
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
def postgres_test_client(postgres_database_manager):
    """Test client using temporary PostgreSQL database - synchronous fixture."""
    from starlette.testclient import TestClient
    from src.api.scraping_api import app
    
    # Patch the scraper service getter to use our test database
    with patch('src.api.scraping_api.get_scraper_service') as mock_service_getter:
        mock_service = Mock()
        mock_service.db = postgres_database_manager
        # Mock other service methods to avoid actual scraping
        mock_service.start_scraping_job = AsyncMock(return_value=TEST_JOB_ID)
        mock_service.get_job_status = Mock(return_value={
            "job_id": TEST_JOB_ID,
            "status": "completed",
            "current_page": 5,
            "total_pages": 5,
            "products_found": 25
        })
        mock_service.stop_scraping_job = AsyncMock(return_value=True)
        mock_service.list_jobs = Mock(return_value=[])
        
        mock_service_getter.return_value = mock_service
        
        client = TestClient(app)
        yield client


@pytest.fixture
def ci_test_client(ci_database_manager):
    """Test client for CI environment - synchronous fixture."""
    from starlette.testclient import TestClient
    from src.api.scraping_api import app
    
    # Patch the scraper service getter to use our CI database
    with patch('src.api.scraping_api.get_scraper_service') as mock_service_getter:
        mock_service = Mock()
        mock_service.db = ci_database_manager
        # Mock other service methods to avoid actual scraping
        mock_service.start_scraping_job = AsyncMock(return_value=TEST_JOB_ID)
        mock_service.get_job_status = Mock(return_value={
            "job_id": TEST_JOB_ID,
            "status": "completed",
            "current_page": 5,
            "total_pages": 5,
            "products_found": 25
        })
        mock_service.stop_scraping_job = AsyncMock(return_value=True)
        mock_service.list_jobs = Mock(return_value=[])
        
        mock_service_getter.return_value = mock_service
        
        client = TestClient(app)
        yield client


@pytest.fixture
def scraper_service_with_mocks(mock_database_manager):
    """ScraperService instance with mocked dependencies."""
    with patch('src.services.scraper_service.DatabaseManager', return_value=mock_database_manager):
        with patch('src.services.scraper_service.Utils') as mock_utils:
            service = ScraperService()
            service.db = mock_database_manager
            return service


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