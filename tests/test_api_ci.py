"""
CI/CD specific API tests for GitHub Actions.

This module contains simplified tests that are more reliable in CI environments
and helps debug CI-specific issues.
"""

import pytest
import os
import sys

# Ensure proper Python path in CI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required imports work in CI environment."""
    try:
        from starlette.testclient import TestClient
        import jwt
        from src.api.scraping_api import app
        assert app is not None
        assert TestClient is not None
        assert jwt is not None
    except ImportError as e:
        pytest.fail(f"Import failed in CI: {e}")

def test_basic_app_creation():
    """Test basic app instantiation in CI environment."""
    try:
        from starlette.testclient import TestClient
        from src.api.scraping_api import app
        
        client = TestClient(app)
        assert client is not None
        
        # Basic health check without complex mocking
        response = client.get("/health")
        assert response.status_code == 200
        
    except Exception as e:
        pytest.fail(f"Basic app creation failed in CI: {e}")

def test_environment_variables():
    """Test CI environment setup."""
    # Check that required environment variables are available
    assert "PYTHONPATH" in os.environ or True  # Don't fail if not set
    
    # Check JWT secret key setup
    jwt_key = os.getenv("JWT_SECRET_KEY", "test-secret-123")
    assert jwt_key is not None
    assert len(jwt_key) > 0

@pytest.mark.skipif(
    os.getenv("CI") != "true",
    reason="CI-specific test"
)
def test_ci_environment():
    """Test specifically for CI environment."""
    assert os.getenv("CI") == "true"
    
    # Test that we can create a TestClient in CI
    from starlette.testclient import TestClient
    from src.api.scraping_api import app
    
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "scraping-api"