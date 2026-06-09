"""FastAPI endpoints for scraping service"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from src.services.scraper_service import ScraperService
from typing import Optional
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
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["mydomain.example", "localhost"])

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "test-secret-123") # secret key ALERT!!!!!!!!!!!!!!! need to change in production, this only for testing purpose, USE .env variable later on

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")
scraper_service = ScraperService()

class ScrapingRequest(BaseModel):
    site: str
    query: str
    max_pages: Optional[int] = 5

class ScrapingResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.post("/api/scraping/start", response_model=ScrapingResponse)
async def start_scraping(request: ScrapingRequest, user_id: str = Depends(verify_token)):
    """Start a new scraping job"""
    try:
        job_id = await scraper_service.start_scraping_job(
            request.site, 
            request.query, 
            request.max_pages
        )
        
        return ScrapingResponse(
            job_id=job_id,
            status="started",
            message="Scraping job started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scraping/status/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(verify_token)):
    """Get scraping job status"""
    status = scraper_service.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@app.post("/api/scraping/stop/{job_id}")
async def stop_scraping_job(job_id: str, user_id: str = Depends(verify_token)):
    """Stop a running scraping job"""
    try:
        result = await scraper_service.stop_scraping_job(job_id)
        
        if result:
            return {"job_id": job_id, "status": "stopped", "message": "Job stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scraping/jobs")
async def list_jobs(user_id: str = Depends(verify_token)):
    """List all scraping jobs"""
    jobs = scraper_service.list_jobs()
    return {"jobs": jobs}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "scraping-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)