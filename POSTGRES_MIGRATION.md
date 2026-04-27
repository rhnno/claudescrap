# PostgreSQL Migration Guide

## Migration Summary

The claudescrap project has been successfully migrated from SQLite to PostgreSQL. This change provides better scalability, concurrent access support, and production readiness.

## What Changed

### 1. Database Configuration
- **Before**: SQLite with `sqlite:///scraping.db`
- **After**: PostgreSQL via `DATABASE_URL` environment variable
- **Requirements**: Must set `DATABASE_URL` for production use

### 2. Requirements Updated
- Added `psycopg2-binary==2.9.7` for PostgreSQL driver
- Added `testing.postgresql>=1.3.0` for test environments
- Removed MongoDB dependency (`pymongo`)

### 3. CI/CD Configuration
- GitLab CI now includes PostgreSQL 13 service
- Automatic database creation and testing
- System dependencies installed automatically

### 4. Test Infrastructure
- New PostgreSQL integration tests in `tests/test_postgres_integration.py`
- Temporary PostgreSQL instances for testing
- CI-specific test fixtures

## Setup Instructions

### Local Development

1. **Install PostgreSQL** (if not already installed):
   ```bash
   # Windows (using Chocolatey)
   choco install postgresql
   
   # macOS (using Homebrew)
   brew install postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   ```

2. **Create Database**:
   ```bash
   # Connect to PostgreSQL
   sudo -u postgres psql
   
   # Create database and user
   CREATE DATABASE scraping_db;
   CREATE USER scraper_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE scraping_db TO scraper_user;
   \q
   ```

3. **Set Environment Variable**:
   ```bash
   # Linux/macOS
   export DATABASE_URL="postgresql://scraper_user:your_password@localhost:5432/scraping_db"
   
   # Windows (PowerShell)
   $env:DATABASE_URL="postgresql://scraper_user:your_password@localhost:5432/scraping_db"
   
   # Windows (Command Prompt)
   set DATABASE_URL=postgresql://scraper_user:your_password@localhost:5432/scraping_db
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_test.txt  # For testing
   ```

### Production Deployment

1. **Set Environment Variables**:
   ```bash
   export DATABASE_URL="postgresql://username:password@host:port/database"
   export JWT_SECRET_KEY="your-production-secret-key"
   ```

2. **Run Application**:
   ```bash
   python run_api.py
   ```

### Testing

1. **Local Testing** (uses temporary PostgreSQL):
   ```bash
   # Run PostgreSQL integration tests
   python tests/test_runner.py --postgres
   
   # Run all tests
   python tests/test_runner.py --all
   ```

2. **CI Testing** (uses GitLab PostgreSQL service):
   ```bash
   # Runs automatically in GitLab CI with PostgreSQL service
   python tests/test_runner.py --ci
   ```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/db` |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | `your-secret-key-here` |

## Troubleshooting

### Common Issues

1. **"DATABASE_URL environment variable is required"**
   - Solution: Set the `DATABASE_URL` environment variable

2. **"testing.postgresql not available"**
   - Solution: Install test dependencies: `pip install -r requirements_test.txt`

3. **PostgreSQL connection refused**
   - Check if PostgreSQL is running: `sudo service postgresql status`
   - Verify connection parameters in `DATABASE_URL`

4. **Permission denied for database**
   - Grant proper privileges to your database user
   - Check user permissions: `\du` in psql

### Validation

Run the migration validation script to check everything is configured correctly:
```bash
python validate_postgres_migration.py
```

## Database Schema

The application automatically creates these tables:
- `scraping_jobs`: Job tracking and status
- `products`: Scraped product data

Schema is created automatically on first run via SQLAlchemy migrations.

## Performance Benefits

- **Concurrent Access**: Multiple workers can access database simultaneously
- **ACID Compliance**: Full transaction support
- **Scalability**: Better performance with large datasets
- **Production Ready**: Suitable for production deployments
- **Advanced Features**: JSON columns, full-text search, etc.

## Migration Validation

All migration changes have been validated:
- ✅ Database models work with PostgreSQL
- ✅ Environment detection logic
- ✅ CI configuration with PostgreSQL service
- ✅ Requirements include PostgreSQL dependencies
- ✅ Test infrastructure supports PostgreSQL

The migration is complete and ready for production use!