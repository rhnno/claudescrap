# Scraping Project Agent Guide

## Project Status: Phase 1 - Core Refactoring (50% Complete)

### ✅ Completed Tasks

#### 1. Separated Concerns
- **Database Layer**: `src/models/database.py` - SQLAlchemy models (ScrapingJob, Product)
- **Service Layer**: `src/services/scraper_service.py` - Business logic separated from presentation
- **API Layer**: `src/api/scraping_api.py` - FastAPI REST endpoints
- **Utils**: Browser management and utilities maintained

#### 2. Async Job Processing
- **Background Tasks**: Implemented with `asyncio.create_task()`
- **Job Tracking**: Running jobs stored in memory dictionary
- **Status Updates**: Real-time job status in database
- **Job Control**: Start, stop, and list job endpoints

#### 3. Database Layer
- **Models**: ScrapingJob and Product tables with relationships
- **Manager**: DatabaseManager with session handling
- **Operations**: Create, update, query jobs and products
- **Storage**: SQLite default, PostgreSQL ready

#### 4. Security Implementation
- **Authentication**: JWT Bearer token authentication
- **CORS**: Configured for specific domains
- **Host Protection**: TrustedHostMiddleware
- **Input Validation**: Pydantic models

### 🔄 Current Architecture

```
src/
├── api/
│   └── scraping_api.py          # FastAPI endpoints with auth
├── models/
│   └── database.py              # SQLAlchemy models + manager
├── services/
│   ├── scraper_service.py       # Core scraping logic (ACTIVE)
│   └── advanced_scraper_service.py  # Enhanced version (NOT USED)
├── utils/
│   ├── browser.py               # Browser management
│   └── utils.py                 # Random utilities
└── ace.py                       # Original monolithic scraper
```

### 🚀 API Endpoints

```
POST /api/scraping/start         # Start scraping job
GET  /api/scraping/status/{id}   # Get job status
POST /api/scraping/stop/{id}     # Stop running job
GET  /api/scraping/jobs          # List all jobs
GET  /health                     # Health check
```

### 🔧 Current Issues & Next Steps

#### Issues to Fix:
1. **Browser Setup**: `setup_browser()` vs `setup_driver()` method mismatch
2. **Import Errors**: Some database model imports need fixing
3. **Error Handling**: Basic error handling, needs improvement
4. **Resource Management**: No browser cleanup on job completion

#### Phase 1 Remaining (50%):
1. **Fix Browser Integration**: Resolve method name mismatches
2. **Improve Error Handling**: Better exception management
3. **Resource Cleanup**: Proper browser session management
4. **Testing**: Unit tests for core components
5. **Documentation**: API documentation with examples

### 📋 Phase 2 Preview: API Development
- WebSocket for real-time updates
- Rate limiting and throttling
- Advanced authentication (roles, permissions)
- Metrics and monitoring endpoints
- Bulk operations support

### 📋 Phase 3 Preview: Dashboard UI
- React/Vue.js frontend
- Real-time job monitoring
- Configuration management UI
- Data visualization and exports
- User management interface

### 🛠️ Development Guidelines

#### File Organization:
- **Models**: Database schemas and relationships
- **Services**: Business logic, no direct HTTP handling
- **API**: HTTP endpoints, authentication, validation
- **Utils**: Shared utilities and helpers

#### Code Standards:
- **Async/Await**: Use async for I/O operations
- **Type Hints**: Add type annotations
- **Error Handling**: Proper exception handling with logging
- **Documentation**: Docstrings for all public methods

#### Security Requirements:
- **Authentication**: JWT required for all scraping endpoints
- **CORS**: Restrict to known domains
- **Input Validation**: Validate all user inputs
- **Rate Limiting**: Implement in Phase 2

### 🔍 Current Configuration

#### Database:
- **Default**: SQLite (`scraping.db`)
- **Production**: Set `DATABASE_URL` environment variable
- **Models**: Auto-created on first run

#### Authentication:
- **JWT Secret**: Set `JWT_SECRET_KEY` environment variable
- **Algorithm**: HS256
- **Required**: All endpoints except `/health`

#### Browser:
- **Manager**: `BrowserManager` class
- **Profile**: Chrome research profile for stealth
- **Cleanup**: Manual cleanup required (needs improvement)

### 📝 Usage Examples

#### Start API:
```bash
export JWT_SECRET_KEY="your-secret-key"
python run_api.py
```

#### Generate JWT:
```python
import jwt
token = jwt.encode({"sub": "user123"}, "your-secret-key", algorithm="HS256")
```

#### API Call:
```bash
curl -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/api/scraping/start \
     -d '{"site": "tokopedia", "query": "laptop", "max_pages": 3}'
```

### 🎯 Success Metrics

#### Phase 1 Goals:
- [x] Separated concerns (database, service, API)
- [x] Basic async job processing
- [x] Database persistence
- [x] REST API with authentication
- [ ] Stable browser integration
- [ ] Proper error handling
- [ ] Resource cleanup

#### Target: Complete Phase 1 before moving to Phase 2

---

*Last Updated: Current session - Phase 1 refactoring in progress*