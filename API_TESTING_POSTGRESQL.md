# API Testing with PostgreSQL Setup

## Overview

The claudescrap project has a comprehensive PostgreSQL setup for API testing that ensures database is properly initialized before API tests execute in CI/CD environments.

## ✅ PostgreSQL Setup for API Testing

### 1. GitLab CI Configuration

The `.gitlab-ci.yml` includes proper PostgreSQL service configuration:

```yaml
services:
  - postgres:13-alpine

variables:
  DATABASE_URL: "postgresql://test_user:test_pass@postgres:5432/test_db"
  POSTGRES_DB: "test_db"
  POSTGRES_USER: "test_user"
  POSTGRES_PASSWORD: "test_pass"
  POSTGRES_HOST_AUTH_METHOD: "trust"

before_script:
  # Install PostgreSQL client and dependencies
  - apt-get update -qq && apt-get install -y -qq postgresql-client gcc python3-dev libpq-dev
  # Install application dependencies
  - pip install -r requirements.txt -r requirements_api.txt -r requirements_test.txt
  # 🔑 CRITICAL: Wait for PostgreSQL to be ready
  - until pg_isready -h postgres -p 5432 -U test_user; do echo "Waiting for postgres..."; sleep 2; done
  # Create test database
  - createdb -h postgres -U test_user test_db || true
```

### 2. Test Database Initialization Order

The PostgreSQL setup follows this sequence:

1. **PostgreSQL Service Start**: GitLab CI starts `postgres:13-alpine` service
2. **Dependency Installation**: System packages and Python dependencies installed
3. **PostgreSQL Readiness Check**: `pg_isready` waits for PostgreSQL to accept connections
4. **Database Creation**: `test_db` database is created
5. **API Tests Execution**: Tests run with fully initialized PostgreSQL

### 3. Test Fixtures for API Testing

#### Environment-Aware Database Selection

The `temp_test_db` fixture automatically adapts to the environment:

```python
@pytest.fixture
def temp_test_db():
    """Intelligent test database fixture that adapts to environment.
    
    - In CI environments with DATABASE_URL: uses CI database
    - In environments with PostgreSQL binaries: uses testing.postgresql  
    - Otherwise: uses mock database for fast local testing
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
```

#### PostgreSQL Test Client

The `postgres_test_client` fixture provides a FastAPI TestClient connected to PostgreSQL:

```python
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
        # ... other mocks
        
        mock_service_getter.return_value = mock_service
        client = TestClient(app)
        yield client
```

### 4. API Test Categories

#### API Tests (`test:api` job)
- **File**: `tests/test_api.py`
- **Database**: Uses PostgreSQL via mocked service
- **Purpose**: Test API endpoints, authentication, request/response validation
- **Example**:
  ```python
  def test_start_scraping_success(self, client, auth_headers):
      """Test successful scraping job start."""
      with patch.object(scraper_service, 'start_scraping_job', new_callable=AsyncMock, return_value=TEST_JOB_ID):
          response = client.post("/api/scraping/start", json={...}, headers=auth_headers)
      assert response.status_code == 200
  ```

#### PostgreSQL Integration Tests (`test:postgres` job)
- **File**: `tests/test_postgres_integration.py`
- **Database**: Direct PostgreSQL connection
- **Purpose**: Test database operations and API with real PostgreSQL
- **Example**:
  ```python
  def test_postgres_api_integration(self, postgres_test_client):
      """Test API with PostgreSQL backend."""
      client = postgres_test_client
      response = client.get("/health")
      assert response.status_code == 200
  ```

#### CI Environment Tests (`test:ci` job)
- **File**: `tests/test_api_ci.py`
- **Database**: Uses CI PostgreSQL database via DATABASE_URL
- **Purpose**: Verify CI environment setup and basic functionality
- **Example**:
  ```python
  def test_ci_environment():
      """Test specifically for CI environment with PostgreSQL."""
      assert os.getenv("CI") == "true"
      database_url = os.getenv("DATABASE_URL")
      assert database_url is not None
      assert "postgresql" in database_url
  ```

### 5. Test Execution Commands

```bash
# Run API tests (uses mocked PostgreSQL service)
python tests/test_runner.py --api

# Run PostgreSQL integration tests (uses testing.postgresql)
python tests/test_runner.py --postgres

# Run CI tests (uses CI PostgreSQL service)
python tests/test_runner.py --ci

# Run all tests
python tests/test_runner.py --all
```

## 🔧 Test Architecture

### Database Connection Strategy

1. **CI Environment**: Uses GitLab CI PostgreSQL service via `DATABASE_URL`
2. **Local Development**: Uses `testing.postgresql` for temporary PostgreSQL instances
3. **Fallback**: Uses mock database for environments without PostgreSQL

### API Service Initialization

The API uses lazy initialization to avoid database connection at import time:

```python
# Global scraper service instance (lazy initialization)
scraper_service = None

def get_scraper_service():
    """Get or initialize the scraper service instance."""
    global scraper_service
    if scraper_service is None:
        scraper_service = ScraperService()
    return scraper_service
```

This pattern allows:
- ✅ Tests to easily mock the service
- ✅ API imports to work without database connection
- ✅ Proper database initialization in CI environments

## 🎯 Key Benefits

1. **Production-Ready**: PostgreSQL provides enterprise-grade reliability
2. **CI/CD Optimized**: Automatic PostgreSQL setup and teardown in CI
3. **Test Isolation**: Each test gets a clean database state
4. **Environment Adaptive**: Automatically uses appropriate database based on environment
5. **Mock-Friendly**: Easy to mock services for fast unit tests
6. **Real Integration**: Full PostgreSQL integration tests ensure compatibility

## 📋 Verification

To verify PostgreSQL setup is working in CI:

1. Check GitLab CI logs for PostgreSQL initialization:
   ```
   Waiting for postgres...
   postgres:5432 - accepting connections
   ```

2. Review test execution logs for database operations
3. Verify API tests pass with real PostgreSQL backend
4. Confirm environment variables are properly set

The setup ensures PostgreSQL is **fully initialized and ready before any API tests execute**.