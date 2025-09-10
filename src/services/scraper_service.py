"""Scraper service - separated from orchestrator"""
import asyncio
import uuid
import logging
from datetime import datetime
from src.models.database import DatabaseManager, ScrapingJob
from src.ace import ScrapingOrchestrator, main
from src.utils.utils import RandomUtils as Utils
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScraperService:
    """
    Enhanced scraper service using ace.py architecture
    Provides async job management with robust error handling and resource cleanup
    """
    def __init__(self):
        """Initialize ScraperService with enhanced architecture"""
        self.db = DatabaseManager()
        self.utils = Utils()
        self.running_jobs = {}  # Track running jobs for stopping
        
        # Session statistics for monitoring
        self.session_stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'total_products': 0,
            'total_pages': 0
        }
        
        logger.info("🚀 Enhanced ScraperService initialized")
    
    async def start_scraping_job(self, site, query, max_pages=5):
        """Start async scraping job with enhanced tracking"""
        job_id = str(uuid.uuid4())
        
        # Update session stats
        self.session_stats['total_jobs'] += 1
        
        # Create job in database
        job = self.db.create_job(job_id, site, query)
        logger.info(f"📋 Created scraping job {job_id} for {site} with query '{query}'")
        
        # Start async scraping and track the task
        task = asyncio.create_task(self._execute_scraping(job_id, site, query, max_pages))
        self.running_jobs[job_id] = {
            'task': task,
            'site': site,
            'query': query,
            'max_pages': max_pages,
            'start_time': datetime.now()
        }
        
        return job_id
    
    async def _execute_scraping(self, job_id, site, query, max_pages):
        """Execute scraping using ScrapingOrchestrator from ace.py"""
        orchestrator = None
        page = 0
        try:
            logger.info(f"Starting scraping job {job_id} for {site} with query '{query}'")
            self.db.update_job_status(job_id, 'running')
            
            # Initialize ScrapingOrchestrator
            try:
                orchestrator = ScrapingOrchestrator()
                if not orchestrator.setup_browser(headless=True, use_profile=True):
                    raise Exception("Browser setup failed")
                logger.info(f"ScrapingOrchestrator initialized successfully for job {job_id}")
            except Exception as e:
                logger.error(f"Failed to initialize ScrapingOrchestrator for job {job_id}: {e}")
                raise Exception(f"ScrapingOrchestrator initialization failed: {e}")
            
            # Use orchestrator's _scrape_query method directly
            try:
                # Check if job was cancelled before starting
                if job_id not in self.running_jobs:
                    logger.info(f"Job {job_id} was cancelled before execution")
                    return
                
                logger.info(f"Using ScrapingOrchestrator to scrape {site} for '{query}' with {max_pages} pages")
                
                # Use ace.py's robust scraping method
                products = orchestrator._scrape_query(
                    site_name=site,
                    query=query,
                    max_pages=max_pages,
                    scroll_depth=3,
                    delay_range=[1, 3]
                )
                
                if products:
                    # Add job metadata to products
                    for product in products:
                        product['job_id'] = job_id
                        product['site'] = site
                        product['query'] = query
                    
                    # Save all products to database
                    try:
                        self.db.save_products(products, job_id)
                        logger.info(f"✅ Saved {len(products)} products to database for job {job_id}")
                    except Exception as e:
                        logger.error(f"Failed to save products for job {job_id}: {e}")
                        raise
                    
                    # Update final page count from orchestrator stats
                    page = orchestrator.session_stats.get('total_pages', max_pages)
                    
                    logger.info(f"✅ ScrapingOrchestrator completed job {job_id}: {len(products)} products from {page} pages")
                else:
                    logger.warning(f"⚠️ No products found for job {job_id}")
                    page = 1
                
                all_products = products or []
                
            except Exception as e:
                logger.error(f"❌ ScrapingOrchestrator failed for job {job_id}: {e}")
                raise
            
            # Complete job successfully
            logger.info(f"✅ Completed scraping job {job_id}: {len(all_products)} total products from {page} pages")
            self.db.update_job_status(
                job_id, 
                'completed', 
                completed_at=datetime.timezone.utc(),
                total_pages=page,
                products_found=len(all_products)
            )
            
            # Update session statistics
            self.session_stats['successful_jobs'] += 1
            self.session_stats['total_products'] += len(all_products)
            self.session_stats['total_pages'] += page
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Job {job_id} was cancelled")
            self.db.update_job_status(
                job_id, 
                'cancelled', 
                completed_at=datetime.now(),
                error_message="Job was cancelled by user"
            )
            self.session_stats['failed_jobs'] += 1
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed with error: {e}")
            self.db.update_job_status(
                job_id, 
                'failed', 
                error_message=str(e),
                completed_at=datetime.timezone.utc(),
                total_pages=page
            )
            self.session_stats['failed_jobs'] += 1
        finally:
            # Ensure orchestrator cleanup
            if orchestrator:
                try:
                    orchestrator.close()
                    logger.info(f"🔧 ScrapingOrchestrator closed for job {job_id}")
                except Exception as e:
                    logger.error(f"❌ Error closing ScrapingOrchestrator for job {job_id}: {e}")
            
            # Remove from running jobs (enhanced tracking)
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
                logger.debug(f"🗑️ Removed job {job_id} from running jobs")
    
    # Note: URL building, product extraction, and scrolling now handled by ScrapingOrchestrator
    
    def get_job_status(self, job_id):
        """Get job status from database"""
        session = self.db.get_session()
        try:
            job = session.query(ScrapingJob).filter(ScrapingJob.job_id == job_id).first()
            if job:
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'current_page': job.current_page,
                    'total_pages': job.total_pages,
                    'products_found': job.products_found,
                    'created_at': job.created_at,
                    'error_message': job.error_message
                }
            return None
        finally:
            session.close()
    
    async def stop_scraping_job(self, job_id):
        """Stop a running scraping job with enhanced tracking"""
        if job_id in self.running_jobs:
            job_info = self.running_jobs[job_id]
            task = job_info['task']
            
            logger.info(f"🛑 Stopping job {job_id} for {job_info['site']} - '{job_info['query']}'")
            task.cancel()
            
            # Update job status in database
            self.db.update_job_status(
                job_id, 
                'stopped', 
                completed_at=datetime.timezone.utc(),
                error_message="Job stopped by user"
            )
            
            # Update session stats
            self.session_stats['failed_jobs'] += 1
            
            # Remove from running jobs
            del self.running_jobs[job_id]
            logger.info(f"✅ Job {job_id} stopped successfully")
            return True
        
        logger.warning(f"⚠️ Job {job_id} not found in running jobs")
        return False
    
    def list_jobs(self):
        """List all scraping jobs"""
        session = self.db.get_session()
        try:
            jobs = session.query(ScrapingJob).order_by(ScrapingJob.created_at.desc()).limit(50).all()
            return [{
                'job_id': job.job_id,
                'status': job.status,
                'site': job.site,
                'query': job.query,
                'current_page': job.current_page,
                'total_pages': job.total_pages,
                'products_found': job.products_found,
                'created_at': job.created_at,
                'completed_at': job.completed_at,
                'error_message': job.error_message
            } for job in jobs]
        finally:
            session.close()
    
    def get_session_stats(self):
        """Get session statistics like ace.py"""
        return {
            'total_jobs': self.session_stats['total_jobs'],
            'successful_jobs': self.session_stats['successful_jobs'],
            'failed_jobs': self.session_stats['failed_jobs'],
            'running_jobs': len(self.running_jobs),
            'total_products': self.session_stats['total_products'],
            'total_pages': self.session_stats['total_pages'],
            'success_rate': (self.session_stats['successful_jobs'] / max(1, self.session_stats['total_jobs'])) * 100
        }
    
    def print_session_summary(self):
        """Print session summary like ace.py"""
        stats = self.get_session_stats()
        
        logger.info("📊 SCRAPER SERVICE SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total jobs: {stats['total_jobs']}")
        logger.info(f"Successful jobs: {stats['successful_jobs']}")
        logger.info(f"Failed jobs: {stats['failed_jobs']}")
        logger.info(f"Currently running: {stats['running_jobs']}")
        logger.info(f"Total products scraped: {stats['total_products']}")
        logger.info(f"Total pages processed: {stats['total_pages']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['total_pages'] > 0:
            avg_products = stats['total_products'] / stats['total_pages']
            logger.info(f"Average products per page: {avg_products:.1f}")