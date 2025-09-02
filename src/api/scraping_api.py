"""FastAPI endpoints for scraping service"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.scraper_service import ScraperService
from typing import Optional

app = FastAPI(title="Scraping API", version="1.0.0")
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
async def start_scraping(request: ScrapingRequest):
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
async def get_job_status(job_id: str):
    """Get scraping job status"""
    status = scraper_service.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@app.get("/api/scraping/jobs")
async def list_jobs():
    """List all scraping jobs"""
    # Implementation for listing jobs
    return {"message": "Jobs list endpoint"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "scraping-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)