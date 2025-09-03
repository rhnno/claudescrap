"""FastAPI endpoints for scraping service"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from src.services.scraper_service import ScraperService
from typing import Optional, Dict, Any
import os
import jwt

app = FastAPI(title="Scraping API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mydomain.example", "http://localhost:3000"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted Host Protection
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["mydomain.example", "localhost", "testserver"])

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "test-secret-123")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and extract user ID.
    
    Validates the provided JWT token using the configured secret key
    and extracts the user identifier from the token payload.
    
    Args:
        credentials (HTTPAuthorizationCredentials): Bearer token from request header
    
    Returns:
        str: User ID extracted from token subject claim
    
    Raises:
        HTTPException: 401 status if token is invalid, expired, or malformed
    
    Example:
        >>> # In request header: Authorization: Bearer <jwt_token>
        >>> user_id = verify_token(credentials)
        >>> print(f"Authenticated user: {user_id}")
    
    Note:
        Uses HS256 algorithm for token validation. Token must contain
        'sub' (subject) claim with user identifier.
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        return str(user_id)  # Ensure string type
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

scraper_service = ScraperService()

class ScrapingRequest(BaseModel):
    """Request model for starting scraping jobs.
    
    Attributes:
        site (str): Target site name (e.g., 'tokopedia', 'shopee')
        query (str): Search query string
        max_pages (Optional[int]): Maximum pages to scrape, defaults to 5
    
    Example:
        >>> request = ScrapingRequest(
        ...     site="tokopedia",
        ...     query="laptop gaming",
        ...     max_pages=10
        ... )
    """
    site: str
    query: str
    max_pages: Optional[int] = 5

    def get_max_pages_value(self) -> int:
        """Get the max_pages value as a non-optional int."""
        return self.max_pages if self.max_pages is not None else 5

class ScrapingResponse(BaseModel):
    """Response model for scraping job operations.
    
    Attributes:
        job_id (str): Unique identifier for the scraping job
        status (str): Current job status (e.g., 'started', 'stopped')
        message (str): Human-readable status message
    
    Example:
        >>> response = ScrapingResponse(
        ...     job_id="job_123",
        ...     status="started",
        ...     message="Scraping job started successfully"
        ... )
    """
    job_id: str
    status: str
    message: str

@app.post("/api/scraping/start", response_model=ScrapingResponse)
async def start_scraping(request: ScrapingRequest, user_id: str = Depends(verify_token)) -> ScrapingResponse:
    """Start a new scraping job.
    
    Initiates an asynchronous scraping job for the specified site and query.
    The job runs in the background and can be monitored via status endpoints.
    
    Args:
        request (ScrapingRequest): Scraping configuration including site, query, and max_pages
        user_id (str): Authenticated user ID from JWT token
    
    Returns:
        ScrapingResponse: Job details including unique job_id and status
    
    Raises:
        HTTPException: 401 for authentication errors, 500 for service errors
    
    Example:
        >>> # POST /api/scraping/start
        >>> # Body: {"site": "tokopedia", "query": "laptop", "max_pages": 5}
        >>> # Response: {"job_id": "job_123", "status": "started", ...}
    
    Note:
        Requires valid JWT token in Authorization header.
        Job execution is asynchronous and non-blocking.
    """
    try:
        job_id = await scraper_service.start_scraping_job(
            request.site, 
            request.query, 
            request.get_max_pages_value()
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to create scraping job")
        
        return ScrapingResponse(
            job_id=job_id,
            status="started",
            message="Scraping job started successfully"
        )
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scraping/status/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(verify_token)) -> Dict[str, Any]:
    """Get scraping job status and progress information.
    
    Retrieves detailed status information for a specific scraping job
    including progress metrics and current state.
    
    Args:
        job_id (str): Unique identifier of the scraping job
        user_id (str): Authenticated user ID from JWT token
    
    Returns:
        Dict[str, Any]: Job status information including:
            - job_id: Job identifier
            - status: Current status (running, completed, failed)
            - current_page: Current page being processed
            - total_pages: Total pages to process
            - products_found: Number of products extracted
    
    Raises:
        HTTPException: 404 if job not found, 401 for authentication errors
    
    Example:
        >>> # GET /api/scraping/status/job_123
        >>> # Response: {"job_id": "job_123", "status": "running", "current_page": 3, ...}
    
    Note:
        Returns real-time progress information for active jobs.
    """
    status = scraper_service.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@app.post("/api/scraping/stop/{job_id}")
async def stop_scraping_job(job_id: str, user_id: str = Depends(verify_token)) -> Dict[str, str]:
    """Stop a running scraping job.
    
    Gracefully terminates an active scraping job and releases associated resources.
    
    Args:
        job_id (str): Unique identifier of the job to stop
        user_id (str): Authenticated user ID from JWT token
    
    Returns:
        Dict[str, str]: Operation result with job_id, status, and message
    
    Raises:
        HTTPException: 404 if job not found, 500 for service errors
    
    Example:
        >>> # POST /api/scraping/stop/job_123
        >>> # Response: {"job_id": "job_123", "status": "stopped", "message": "..."}
    
    Note:
        Stopping a job may take a few seconds to complete gracefully.
        Already completed jobs cannot be stopped.
    """
    try:
        result = await scraper_service.stop_scraping_job(job_id)
        
        if result:
            return {"job_id": job_id, "status": "stopped", "message": "Job stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
            
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scraping/jobs")
async def list_jobs(user_id: str = Depends(verify_token)) -> Dict[str, Any]:
    """List all scraping jobs for the authenticated user.
    
    Retrieves a comprehensive list of all scraping jobs associated with
    the authenticated user, including their current status.
    
    Args:
        user_id (str): Authenticated user ID from JWT token
    
    Returns:
        Dict[str, Any]: Dictionary containing 'jobs' key with list of job objects
    
    Example:
        >>> # GET /api/scraping/jobs
        >>> # Response: {
        >>> #   "jobs": [
        >>> #     {"job_id": "job_1", "status": "completed", "site": "tokopedia", ...},
        >>> #     {"job_id": "job_2", "status": "running", "site": "shopee", ...}
        >>> #   ]
        >>> # }
    
    Note:
        Returns jobs in reverse chronological order (newest first).
    """
    jobs = scraper_service.list_jobs()
    return {"jobs": jobs}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for service monitoring.
    
    Provides a simple health status check for monitoring systems
    and load balancers to verify service availability.
    
    Returns:
        Dict[str, str]: Health status with service identifier
    
    Example:
        >>> # GET /health
        >>> # Response: {"status": "healthy", "service": "scraping-api"}
    
    Note:
        Does not require authentication. Used by monitoring systems
        to verify service health.
    """
    return {"status": "healthy", "service": "scraping-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)