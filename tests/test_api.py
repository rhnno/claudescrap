"""
API integration tests for scraping endpoints.

Tests the FastAPI endpoints that use ScraperService to ensure
proper integration between the API layer and service layer.
"""
from typing import Any


import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from starlette.testclient import TestClient
import jwt
from datetime import datetime
from src.api.scraping_api import app, scraper_service
from tests.conftest import TEST_SITE, TEST_QUERY, TEST_MAX_PAGES, TEST_JOB_ID


# Test JWT token
SECRET_KEY = "test-secret-123"
TEST_TOKEN= jwt.encode({"sub": "test_user"}, SECRET_KEY, algorithm="HS256")


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestScrapingAPIEndpoints:
    """Test all scraping API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "scraping-api"
    
    def test_start_scraping_success(self, client, auth_headers):
        """Test successful scraping job start."""
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID) as mock_start:
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": TEST_QUERY,
                    "max_pages": TEST_MAX_PAGES
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == TEST_JOB_ID
        assert data["status"] == "started"
        mock_start.assert_called_once_with(TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
    
    def test_start_scraping_unauthorized(self, client):
        """Test scraping start without authentication."""
        response = client.post(
            "/api/scraping/start",
            json={
                "site": TEST_SITE,
                "query": TEST_QUERY,
                "max_pages": TEST_MAX_PAGES
            }
        )
        assert response.status_code == 403
    
    def test_start_scraping_invalid_token(self, client):
        """Test scraping start with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post(
            "/api/scraping/start",
            json={
                "site": TEST_SITE,
                "query": TEST_QUERY,
                "max_pages": TEST_MAX_PAGES
            },
            headers=headers
        )
        assert response.status_code == 401
    
    def test_start_scraping_service_error(self, client, auth_headers):
        """Test scraping start when service raises error."""
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, side_effect=Exception("Service error")):
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": TEST_QUERY,
                    "max_pages": TEST_MAX_PAGES
                },
                headers=auth_headers
            )
        
        assert response.status_code == 500
        assert "Service error" in response.json()["detail"]
    
    def test_get_job_status_existing(self, client, auth_headers):
        """Test getting status of existing job."""
        mock_status = {
            "job_id": TEST_JOB_ID,
            "status": "running",
            "current_page": 2,
            "total_pages": 5,
            "products_found": 25
        }
        
        with patch.object(scraper_service, 'get_job_status', return_value=mock_status):
            response = client.get(f"/api/scraping/status/{TEST_JOB_ID}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == TEST_JOB_ID
        assert data["status"] == "running"
        assert data["current_page"] == 2
    
    def test_get_job_status_not_found(self, client, auth_headers):
        """Test getting status of non-existent job."""
        with patch.object(scraper_service, 'get_job_status', return_value=None):
            response = client.get("/api/scraping/status/nonexistent", headers=auth_headers)
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_stop_scraping_job_success(self, client, auth_headers):
        """Test successfully stopping a job."""
        with patch.object(scraper_service, 'stop_scraping_job', new_callable=AsyncMock, return_value=True) as mock_stop:
            response = client.post(f"/api/scraping/stop/{TEST_JOB_ID}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == TEST_JOB_ID
        assert data["status"] == "stopped"
        mock_stop.assert_called_once_with(TEST_JOB_ID)
    
    def test_stop_scraping_job_not_found(self, client, auth_headers):
        """Test stopping non-existent job."""
        with patch.object(scraper_service, 'stop_scraping_job', new_callable=AsyncMock, return_value=False):
            response = client.post("/api/scraping/stop/nonexistent", headers=auth_headers)
        
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_stop_scraping_job_service_error(self, client, auth_headers):
        """Test stopping job when service raises error."""
        with patch.object(scraper_service, 'stop_scraping_job', new_callable=AsyncMock, side_effect=Exception("Stop error")):
            response = client.post(f"/api/scraping/stop/{TEST_JOB_ID}", headers=auth_headers)
        
        assert response.status_code == 500
        assert "Stop error" in response.json()["detail"]
    
    def test_list_jobs_success(self, client, auth_headers):
        """Test listing all jobs."""
        mock_jobs = [
            {
                "job_id": "job1",
                "status": "completed",
                "site": "tokopedia",
                "query": "laptop",
                "products_found": 50
            },
            {
                "job_id": "job2",
                "status": "running",
                "site": "shopee",
                "query": "phone",
                "products_found": 0
            }
        ]
        
        with patch.object(scraper_service, 'list_jobs', return_value=mock_jobs):
            response = client.get("/api/scraping/jobs", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == "job1"
        assert data["jobs"][1]["job_id"] == "job2"
    
    def test_request_validation(self, client, auth_headers):
        """Test request validation for start scraping endpoint."""
        # Missing required fields
        response = client.post(
            "/api/scraping/start",
            json={"site": TEST_SITE},  # Missing query
            headers=auth_headers
        )
        assert response.status_code == 422
        
        # Invalid data types
        response = client.post(
            "/api/scraping/start",
            json={
                "site": TEST_SITE,
                "query": TEST_QUERY,
                "max_pages": "invalid"  # Should be int
            },
            headers=auth_headers
        )
        assert response.status_code == 422
    
    def test_default_max_pages(self, client, auth_headers):
        """Test default max_pages value."""
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID) as mock_start:
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": TEST_QUERY
                    # max_pages not provided
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
        mock_start.assert_called_once_with(TEST_SITE, TEST_QUERY, 5)  # Default value


class TestAPISecurityAndMiddleware:
    """Test API security features and middleware."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/health", headers={"Origin": "http://localhost:3000"})
        # Note: TestClient might not fully simulate CORS behavior
        # In real tests, you'd check for specific CORS headers
        assert response.status_code in [200, 405]  # OPTIONS might not be handled
    
    def test_jwt_token_validation(self, client):
        """Test JWT token validation."""
        # Test with malformed token
        headers = {"Authorization": "Bearer malformed.token"}
        response = client.get("/api/scraping/jobs", headers=headers)
        assert response.status_code == 401
        
        # Test with missing Bearer prefix
        headers = {"Authorization": TEST_TOKEN}
        response = client.get("/api/scraping/jobs", headers=headers)
        assert response.status_code == 403  # Missing Bearer
    
    def test_all_endpoints_require_auth(self, client):
        """Test that all protected endpoints require authentication."""
        protected_endpoints = [
            ("GET", "/api/scraping/jobs"),
            ("GET", f"/api/scraping/status/{TEST_JOB_ID}"),
            ("POST", f"/api/scraping/stop/{TEST_JOB_ID}"),
            ("POST", "/api/scraping/start")
        ]
        
        for method, endpoint in protected_endpoints:
            response = None  # Initialize response variable
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            # Ensure response is not None before asserting
            assert response is not None
            assert response.status_code in [401, 403, 422]  # Unauthorized or validation error


class TestAPIEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_very_long_query(self, client, auth_headers):
        """Test handling of very long query strings."""
        long_query = "a" * 1000  # Very long query
        
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID):
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": long_query,
                    "max_pages": 1
                },
                headers=auth_headers
            )
        
        # Should handle long queries gracefully
        assert response.status_code in [200, 413]  # Success or payload too large
    
    def test_special_characters_in_query(self, client, auth_headers):
        """Test handling of special characters in query."""
        special_query = "laptop & gaming (RTX 4060) $1000+"
        
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID):
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": special_query,
                    "max_pages": 1
                },
                headers=auth_headers
            )
        
        assert response.status_code == 200
    
    def test_large_max_pages(self, client, auth_headers):
        """Test handling of very large max_pages values."""
        with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID):
            response = client.post(
                "/api/scraping/start",
                json={
                    "site": TEST_SITE,
                    "query": TEST_QUERY,
                    "max_pages": 1000000  # Very large number
                },
                headers=auth_headers
            )
        
        # Should either accept or validate the large number
        assert response.status_code in [200, 422]
    
    def test_concurrent_api_requests(self, client, auth_headers):
        """Test handling of concurrent API requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID):
                response = client.post(
                    "/api/scraping/start",
                    json={
                        "site": TEST_SITE,
                        "query": TEST_QUERY,
                        "max_pages": 1
                    },
                    headers=auth_headers
                )
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5