"""
Scraper Service Module

This module provides the ScraperService class that implements an enhanced web scraping service
using the ace.py architecture. It provides async job management with robust error handling,
resource cleanup, and browser pooling for optimal performance.

The service integrates with ScrapingOrchestrator for proven scraping capabilities and includes
comprehensive job tracking, session statistics, and database persistence.

Example:
    Basic usage of the scraper service::

        service = ScraperService()
        job_id = await service.start_scraping_job('tokopedia', 'laptop', max_pages=5)
        status = service.get_job_status(job_id)
        await service.stop_scraping_job(job_id)

Note:
    This service requires a configured database connection and browser setup.
    Browser pooling is enabled by default for performance optimization.
"""
import asyncio
import uuid
import logging
from datetime import datetime, timezone
from src.models.database import DatabaseManager, ScrapingJob
from src.ace import ScrapingOrchestrator
from src.utils.utils import RandomUtils as Utils
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScraperService:
    """
    Enhanced scraper service using ace.py architecture.
    
    This class provides comprehensive async job management with robust error handling,
    resource cleanup, and browser pooling for optimal performance. It integrates with
    the ScrapingOrchestrator from ace.py for proven scraping capabilities.
    
    Attributes:
        db (DatabaseManager): Database manager instance for job persistence
        utils (RandomUtils): Utility functions for random delays and human-like behavior
        running_jobs (dict): Dictionary tracking currently running scraping jobs
        session_stats (dict): Session statistics including job counts and success rates
        
    Browser Pool Attributes:
        _browser_pool (list): List of available browser instances for reuse
        _browser_pool_size (int): Maximum number of browsers to maintain in pool (default: 2)
        _browser_lock (asyncio.Lock): Async lock for thread-safe browser pool operations
        
    Example:
        Create and use the scraper service::
        
            service = ScraperService()
            
            # Start a scraping job
            job_id = await service.start_scraping_job(
                site='tokopedia',
                query='gaming laptop',
                max_pages=3
            )
            
            # Check job status
            status = service.get_job_status(job_id)
            print(f"Job status: {status['status']}")
            
            # Get session statistics
            stats = service.get_session_stats()
            print(f"Success rate: {stats['success_rate']:.1f}%")
            
    Note:
        - Browser pool is automatically managed for performance optimization
        - All jobs are tracked in the database for persistence
        - Session statistics are maintained for monitoring and analysis
        - Proper resource cleanup is guaranteed even on failures
    """
    def __init__(self):
        """
        Initialize ScraperService with enhanced architecture.
        
        Sets up the database connection, utilities, job tracking, browser pool,
        and session statistics. The browser pool is pre-configured for optimal
        performance with async browser creation and reuse.
        
        Raises:
            Exception: If database initialization fails
            
        Example:
            Initialize the service::
            
                service = ScraperService()
                print(f"Service initialized with {service._browser_pool_size} browser pool")
                
        Note:
            - Browser pool starts empty and creates browsers on-demand
            - Session statistics are reset on each initialization
            - All running jobs are tracked in memory for this session
        """
        self.db = DatabaseManager()
        self.utils = Utils()
        self.running_jobs = {}  # Track running jobs for stopping
        
        # Browser pool for performance optimization
        self._browser_pool = []
        self._browser_pool_size = 2
        self._browser_lock = asyncio.Lock()
        
        # Session statistics for monitoring
        self.session_stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'total_products': 0,
            'total_pages': 0
        }
        
        logger.info("🚀 Enhanced ScraperService initialized with browser pool")
    
    async def _get_browser_from_pool(self) -> ScrapingOrchestrator:
        """
        Get browser from pool or create new one asynchronously.
        
        This method implements the browser pool pattern for performance optimization.
        It first attempts to reuse an available browser from the pool. If no browsers
        are available, it creates a new one asynchronously using thread pool execution.
        
        Returns:
            ScrapingOrchestrator: A ready-to-use browser orchestrator instance
            
        Raises:
            Exception: If browser creation fails or pool operations encounter errors
            
        Example:
            Get a browser for scraping::
            
                orchestrator = await service._get_browser_from_pool()
                # Use orchestrator for scraping operations
                await service._return_browser_to_pool(orchestrator)
                
        Note:
            - This method is thread-safe using async locks
            - Browser creation is performed in a separate thread to avoid blocking
            - Pool statistics are logged for monitoring
        """
        async with self._browser_lock:
            if self._browser_pool:
                orchestrator = self._browser_pool.pop()
                logger.info(f"🔄 Reusing browser from pool ({len(self._browser_pool)} remaining)")
                return orchestrator
            else:
                logger.info("🏭 Creating new browser (pool empty)")
                return await self._create_browser_async()
    
    async def _return_browser_to_pool(self, orchestrator: ScrapingOrchestrator) -> None:
        """
        Return browser to pool for reuse or close if pool is full.
        
        This method implements proper browser lifecycle management by returning
        browsers to the pool for reuse when possible, or safely closing them
        when the pool is at capacity.
        
        Args:
            orchestrator (ScrapingOrchestrator): The browser orchestrator to return
            
        Raises:
            Exception: If browser closure fails (logged but not re-raised)
            
        Example:
            Return a browser after use::
            
                orchestrator = await service._get_browser_from_pool()
                try:
                    # Perform scraping operations
                    products = orchestrator._scrape_query('tokopedia', 'laptop', 3)
                finally:
                    await service._return_browser_to_pool(orchestrator)
                    
        Note:
            - This method is thread-safe using async locks
            - Excess browsers are safely closed when pool is full
            - Pool statistics are logged for monitoring
            - Errors during browser closure are logged but don't propagate
        """
        async with self._browser_lock:
            if len(self._browser_pool) < self._browser_pool_size:
                self._browser_pool.append(orchestrator)
                logger.info(f"🔄 Returned browser to pool ({len(self._browser_pool)} available)")
            else:
                # Pool is full, close this browser
                try:
                    orchestrator.close()
                    logger.info("🗑️ Browser pool full, closed excess browser")
                except Exception as e:
                    logger.error(f"⚠️ Error closing excess browser: {e}")
    
    async def _create_browser_async(self) -> ScrapingOrchestrator:
        """
        Create browser asynchronously in thread pool to avoid blocking.
        
        This method creates a new ScrapingOrchestrator with browser setup in a
        separate thread to prevent blocking the main async event loop. The browser
        is configured with headless mode and user profile for optimal performance.
        
        Returns:
            ScrapingOrchestrator: A fully configured browser orchestrator instance
            
        Raises:
            Exception: If browser setup fails or thread pool execution encounters errors
            
        Example:
            Create a new browser asynchronously::
            
                orchestrator = await service._create_browser_async()
                print(f"Browser created successfully: {orchestrator is not None}")
                
        Note:
            - Browser creation runs in ThreadPoolExecutor to avoid blocking
            - Browser is configured with headless=True and use_profile=True
            - This method is called automatically by _get_browser_from_pool when needed
            - Creation time is logged for performance monitoring
        """
        import concurrent.futures
        
        def _sync_browser_creation():
            orchestrator = ScrapingOrchestrator()
            success = orchestrator.setup_browser(headless=True, use_profile=True)
            if not success:
                raise Exception("Browser setup failed")
            return orchestrator
        
        # Run browser creation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            orchestrator = await loop.run_in_executor(executor, _sync_browser_creation)
            logger.info("✅ Browser created asynchronously")
            return orchestrator
    
    async def start_scraping_job(self, site: str, query: str, max_pages: int = 5) -> str:
        """
        Start an asynchronous scraping job with enhanced tracking.
        
        This method creates a new scraping job, stores it in the database, and
        starts the scraping process in the background. The job is tracked in memory
        for real-time status updates and can be stopped if needed.
        
        Args:
            site (str): The e-commerce site to scrape ('tokopedia' or 'shopee')
            query (str): The search query/product keyword to scrape
            max_pages (int, optional): Maximum number of pages to scrape. Defaults to 5.
            
        Returns:
            str: Unique job ID for tracking the scraping job
            
        Raises:
            Exception: If job creation fails or database operations encounter errors
            
        Example:
            Start a scraping job::
            
                job_id = await service.start_scraping_job(
                    site='tokopedia',
                    query='gaming laptop RTX 4060',
                    max_pages=10
                )
                print(f"Started job: {job_id}")
                
                # Check status later
                status = service.get_job_status(job_id)
                print(f"Job status: {status['status']}")
                
        Note:
            - Job ID is a UUID4 string for uniqueness
            - Job is immediately stored in database with 'pending' status
            - Scraping runs in background using asyncio.create_task
            - Session statistics are updated automatically
            - Job can be stopped using stop_scraping_job method
        """
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
            'start_time': datetime.now(timezone.utc)
        }
        
        return job_id
    
    async def _execute_scraping(self, job_id: str, site: str, query: str, max_pages: int) -> None:
        """
        Execute scraping using ScrapingOrchestrator from ace.py with browser pool.
        
        This is the core scraping execution method that handles the complete scraping
        workflow: browser acquisition, scraping execution, data storage, and cleanup.
        It uses the browser pool for performance optimization and provides comprehensive
        error handling and logging.
        
        Args:
            job_id (str): Unique identifier for the scraping job
            site (str): The e-commerce site to scrape ('tokopedia' or 'shopee')
            query (str): The search query/product keyword to scrape
            max_pages (int): Maximum number of pages to scrape
            
        Raises:
            asyncio.CancelledError: If job is cancelled during execution
            Exception: If browser acquisition, scraping, or data storage fails
            
        Example:
            This method is called internally by start_scraping_job::
            
                # Internal usage - not called directly
                await service._execute_scraping('job-123', 'tokopedia', 'laptop', 5)
                
        Note:
            - Browser is acquired from pool for performance optimization
            - Job status is updated in database throughout execution
            - Products are saved to database with job metadata
            - Session statistics are updated on completion/failure
            - Browser is always returned to pool in finally block
            - Comprehensive logging with emoji indicators for easy monitoring
            - Handles cancellation gracefully when job is stopped
        """
        orchestrator = None
        page = 0
        try:
            logger.info(f"Starting scraping job {job_id} for {site} with query '{query}'")
            self.db.update_job_status(job_id, 'running')
            
            # Get browser from pool (async, much faster!)
            try:
                orchestrator = await self._get_browser_from_pool()
                logger.info(f"ScrapingOrchestrator obtained from pool for job {job_id}")
            except Exception as e:
                logger.error(f"Failed to get browser for job {job_id}: {e}")
                raise Exception(f"Browser acquisition failed: {e}")
            
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
                    if page is None:
                        page = max_pages
                    
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
                completed_at=datetime.now(timezone.utc),
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
                completed_at=datetime.now(timezone.utc),
                error_message="Job was cancelled by user"
            )
            self.session_stats['failed_jobs'] += 1
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed with error: {e}")
            self.db.update_job_status(
                job_id, 
                'failed', 
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
                total_pages=page
            )
            self.session_stats['failed_jobs'] += 1
        finally:
            # Return orchestrator to pool instead of closing
            if orchestrator:
                try:
                    await self._return_browser_to_pool(orchestrator)
                    logger.info(f"🔄 Browser returned to pool for job {job_id}")
                except Exception as e:
                    logger.error(f"⚠️ Error returning browser to pool for job {job_id}: {e}")
                    # Fallback: close browser if pool return fails
                    try:
                        orchestrator.close()
                    except:
                        pass
            
            # Remove from running jobs (enhanced tracking)
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
                logger.debug(f"🗑️ Removed job {job_id} from running jobs")
    
    # Note: URL building, product extraction, and scrolling now handled by ScrapingOrchestrator
    
    def get_job_status(self, job_id: str) -> dict | None:
        """
        Get comprehensive job status from database.
        
        Retrieves the current status and details of a scraping job from the database.
        Returns detailed information including progress, results, and any error messages.
        
        Args:
            job_id (str): Unique identifier for the scraping job
            
        Returns:
            dict: Job status information containing:
                - job_id (str): The job identifier
                - status (str): Current job status ('pending', 'running', 'completed', 'failed', 'cancelled', 'stopped')
                - current_page (int): Current page being processed
                - total_pages (int): Total pages processed
                - products_found (int): Number of products found
                - created_at (datetime): Job creation timestamp
                - error_message (str): Error message if job failed
                
            None: If job with the given ID is not found
            
        Example:
            Check job status::
            
                status = service.get_job_status('job-123')
                if status:
                    print(f"Status: {status['status']}")
                    print(f"Progress: {status['current_page']}/{status['total_pages']}")
                    print(f"Products found: {status['products_found']}")
                else:
                    print("Job not found")
                    
        Note:
            - Database session is properly managed with try/finally
            - Returns None if job doesn't exist
            - All timestamps are in UTC
            - Status values correspond to job lifecycle states
        """
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
    
    async def stop_scraping_job(self, job_id: str) -> bool:
        """
        Stop a running scraping job with enhanced tracking.
        
        Gracefully stops a running scraping job by cancelling its async task and
        updating the job status in the database. The browser used by the job is
        properly returned to the pool for reuse.
        
        Args:
            job_id (str): Unique identifier for the scraping job to stop
            
        Returns:
            bool: True if job was successfully stopped, False if job was not found
            
        Example:
            Stop a running job::
            
                success = await service.stop_scraping_job('job-123')
                if success:
                    print("Job stopped successfully")
                else:
                    print("Job not found or already completed")
                    
                # Check final status
                status = service.get_job_status('job-123')
                print(f"Final status: {status['status']}")
                
        Note:
            - Only affects jobs that are currently running
            - Job status is updated to 'stopped' in database
            - Session statistics are updated to reflect the stopped job
            - Browser resources are properly cleaned up
            - Task cancellation is handled gracefully by _execute_scraping
        """
        if job_id in self.running_jobs:
            job_info = self.running_jobs[job_id]
            task = job_info['task']
            
            logger.info(f"🛑 Stopping job {job_id} for {job_info['site']} - '{job_info['query']}'")
            task.cancel()
            
            # Update job status in database
            self.db.update_job_status(
                job_id, 
                'stopped', 
                completed_at=datetime.now(timezone.utc),
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
    
    def list_jobs(self) -> list:
        """
        List all scraping jobs with comprehensive details.
        
        Retrieves a list of all scraping jobs from the database, ordered by creation
        time (newest first). Limited to the most recent 50 jobs for performance.
        
        Returns:
            list: List of job dictionaries, each containing:
                - job_id (str): Unique job identifier
                - status (str): Current job status
                - site (str): E-commerce site scraped
                - query (str): Search query used
                - current_page (int): Current/final page processed
                - total_pages (int): Total pages processed
                - products_found (int): Number of products found
                - created_at (datetime): Job creation timestamp
                - completed_at (datetime): Job completion timestamp (if completed)
                - error_message (str): Error message (if failed)
                
        Example:
            List and display all jobs::
            
                jobs = service.list_jobs()
                print(f"Found {len(jobs)} jobs:")
                
                for job in jobs:
                    print(f"ID: {job['job_id'][:8]}...")
                    print(f"Site: {job['site']}, Query: {job['query']}")
                    print(f"Status: {job['status']}, Products: {job['products_found']}")
                    print(f"Created: {job['created_at']}")
                    print("-" * 40)
                    
        Note:
            - Results are ordered by creation time (newest first)
            - Limited to 50 most recent jobs for performance
            - Database session is properly managed
            - All timestamps are in UTC
            - Includes both completed and active jobs
        """
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
    
    def get_session_stats(self) -> dict:
        """
        Get comprehensive session statistics like ace.py.
        
        Returns detailed statistics about the current scraping session including
        job counts, success rates, and performance metrics. Statistics are maintained
        in memory and reset when the service is reinitialized.
        
        Returns:
            dict: Session statistics containing:
                - total_jobs (int): Total number of jobs started
                - successful_jobs (int): Number of successfully completed jobs
                - failed_jobs (int): Number of failed/stopped/cancelled jobs
                - running_jobs (int): Number of currently running jobs
                - total_products (int): Total products scraped across all jobs
                - total_pages (int): Total pages processed across all jobs
                - success_rate (float): Success rate percentage (0-100)
                
        Example:
            Get and display session statistics::
            
                stats = service.get_session_stats()
                print(f"Jobs: {stats['successful_jobs']}/{stats['total_jobs']} successful")
                print(f"Success rate: {stats['success_rate']:.1f}%")
                print(f"Products scraped: {stats['total_products']}")
                print(f"Pages processed: {stats['total_pages']}")
                
                if stats['total_pages'] > 0:
                    avg_products = stats['total_products'] / stats['total_pages']
                    print(f"Average products per page: {avg_products:.1f}")
                    
        Note:
            - Statistics are maintained in memory for the current session
            - Success rate is calculated as successful_jobs / total_jobs * 100
            - Running jobs count reflects real-time active jobs
            - Statistics are updated automatically as jobs complete
        """
        return {
            'total_jobs': self.session_stats['total_jobs'],
            'successful_jobs': self.session_stats['successful_jobs'],
            'failed_jobs': self.session_stats['failed_jobs'],
            'running_jobs': len(self.running_jobs),
            'total_products': self.session_stats['total_products'],
            'total_pages': self.session_stats['total_pages'],
            'success_rate': (self.session_stats['successful_jobs'] / max(1, self.session_stats['total_jobs'])) * 100
        }
    
    def print_session_summary(self) -> None:
        """
        Print comprehensive session summary like ace.py.
        
        Displays a formatted summary of the current scraping session statistics
        including job counts, success rates, and performance metrics. Uses logging
        for consistent output formatting with emoji indicators.
        
        Example:
            Print session summary::
            
                service.print_session_summary()
                
                # Output:
                # 📊 SCRAPER SERVICE SESSION SUMMARY
                # ==================================================
                # Total jobs: 15
                # Successful jobs: 12
                # Failed jobs: 3
                # Currently running: 2
                # Total products scraped: 1,247
                # Total pages processed: 45
                # Success rate: 80.0%
                # Average products per page: 27.7
                
        Note:
            - Uses logger.info for consistent formatting
            - Includes emoji indicators for visual clarity
            - Calculates and displays average products per page if applicable
            - Summary includes both historical and real-time data
            - Automatically called by some CLI tools for session reporting
        """
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