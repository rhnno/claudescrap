"""
PostgreSQL integration tests.

These tests use a temporary PostgreSQL server to ensure proper
database integration and SQL compatibility without requiring
external PostgreSQL services.
"""
import pytest
import os

# Skip all tests if testing.postgresql not available
testing_postgresql = pytest.importorskip("testing.postgresql")

from src.models.database import DatabaseManager, ScrapingJob, Product
from tests.conftest import TEST_JOB_ID, auth_headers


class TestPostgreSQLIntegration:
    """Test database operations with temporary PostgreSQL server."""
    
    def test_postgres_connection(self, postgres_database_manager):
        """Test PostgreSQL database connection."""
        db = postgres_database_manager
        session = db.get_session()
        try:
            # Test basic connectivity with PostgreSQL-specific query
            result = session.execute("SELECT version()").fetchone()
            assert result is not None
            assert "PostgreSQL" in str(result[0])
        finally:
            session.close()
    
    def test_postgres_job_operations(self, postgres_database_manager):
        """Test job CRUD operations with PostgreSQL."""
        db = postgres_database_manager
        
        # Create a job
        job = db.create_job("postgres-job-123", "tokopedia", "gaming setup")
        assert job.job_id == "postgres-job-123"
        
        # Update job status
        db.update_job_status(
            "postgres-job-123",
            "running", 
            current_page=2,
            total_pages=8,
            products_found=15
        )
        
        # Verify updates
        session = db.get_session()
        try:
            updated_job = session.query(ScrapingJob).filter(
                ScrapingJob.job_id == "postgres-job-123"
            ).first()
            
            assert updated_job.status == "running"
            assert updated_job.current_page == 2
            assert updated_job.total_pages == 8
            assert updated_job.products_found == 15
        finally:
            session.close()
    
    def test_postgres_product_operations(self, postgres_database_manager):
        """Test product CRUD operations with PostgreSQL."""
        db = postgres_database_manager
        
        # Create a job first
        job = db.create_job("product-test-job", "shopee", "smartphone")
        
        # Sample products
        products = [
            {
                "name": "iPhone 15 Pro",
                "price": "Rp 18.999.000", 
                "url": "https://shopee.co.id/iphone-15-pro",
                "site": "shopee",
                "query": "smartphone",
                "page_number": 1
            },
            {
                "name": "Samsung Galaxy S24",
                "price": "Rp 15.999.000",
                "url": "https://shopee.co.id/samsung-s24", 
                "site": "shopee",
                "query": "smartphone",
                "page_number": 1
            }
        ]
        
        # Save products
        db.save_products(products, "product-test-job")
        
        # Retrieve and verify products
        session = db.get_session()
        try:
            saved_products = session.query(Product).filter(
                Product.job_id == "product-test-job"
            ).all()
            
            assert len(saved_products) == 2
            assert saved_products[0].name in ["iPhone 15 Pro", "Samsung Galaxy S24"]
            assert saved_products[1].name in ["iPhone 15 Pro", "Samsung Galaxy S24"]
            assert all(p.job_id == "product-test-job" for p in saved_products)
        finally:
            session.close()
    
    def test_postgres_api_integration(self, postgres_test_client):
        """Test API with PostgreSQL backend."""
        client = postgres_test_client
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Get auth headers
        import jwt
        SECRET_KEY = "test-secret-123"
        test_token = jwt.encode({"sub": "test_user"}, SECRET_KEY, algorithm="HS256")
        headers = {"Authorization": f"Bearer {test_token}"}
        
        # Test scraping start endpoint
        response = client.post(
            "/api/scraping/start",
            json={
                "site": "shopee",
                "query": "wireless headphones", 
                "max_pages": 3
            },
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"


# CI-specific tests that run with PostgreSQL in CI environment
class TestCIEnvironment:
    """Tests designed to run reliably in CI environment with PostgreSQL."""
    
    def test_ci_database_connection(self, ci_database_manager):
        """Test PostgreSQL database connection in CI environment."""
        db = ci_database_manager
        session = db.get_session()
        try:
            # Test PostgreSQL connectivity
            result = session.execute("SELECT current_database()").fetchone()
            assert result is not None
            assert result[0] == "test_db"  # CI database name
        finally:
            session.close()
    
    def test_ci_api_health(self, ci_test_client):
        """Test API health check in CI environment."""
        client = ci_test_client
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "scraping-api"
    
    def test_ci_database_operations(self, ci_database_manager):
        """Test database operations in CI PostgreSQL environment."""
        db = ci_database_manager
        
        # Create a job
        job = db.create_job("ci-test-job", "tokopedia", "ci testing")
        assert job.job_id == "ci-test-job"
        assert job.site == "tokopedia"
        assert job.query == "ci testing"
        
        # Update job
        db.update_job_status("ci-test-job", "completed", products_found=10)
        
        # Verify update
        session = db.get_session()
        try:
            updated_job = session.query(ScrapingJob).filter(
                ScrapingJob.job_id == "ci-test-job"
            ).first()
            
            assert updated_job.status == "completed"
            assert updated_job.products_found == 10
        finally:
            session.close()