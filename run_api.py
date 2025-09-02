"""Run the scraping API server"""
import uvicorn
from src.api.scraping_api import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api.scraping_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )