"""Scraper service - separated from orchestrator"""
import asyncio
import uuid
from datetime import datetime
from src.models.database import DatabaseManager
from src.utils.browser import BrowserManager
from src.utils.utils import RandomUtils as Utils
from src.models.database import ScrapingJob
import time

class ScraperService:
    def __init__(self):
        self.db = DatabaseManager()
        self.utils = Utils()
        self.running_jobs = {}  # Track running jobs for stopping
    
    async def start_scraping_job(self, site, query, max_pages=5):
        """Start async scraping job"""
        job_id = str(uuid.uuid4())
        
        # Create job in database
        job = self.db.create_job(job_id, site, query)
        
        # Start async scraping and track the task
        task = asyncio.create_task(self._execute_scraping(job_id, site, query, max_pages))
        self.running_jobs[job_id] = task
        
        return job_id
    
    async def _execute_scraping(self, job_id, site, query, max_pages):
        """Execute scraping in background"""
        try:
            self.db.update_job_status(job_id, 'running')
            
            browser = BrowserManager()
            browser.setup_driver()
            
            all_products = []
            
            for page in range(1, max_pages + 1):
                self.db.update_job_status(job_id, 'running', current_page=page)
                
                # Build URL
                url = self._build_url(site, query, page)
                browser.navigate_to(url)
                
                # Wait for page load
                await asyncio.sleep(2)
                
                # Extract products
                products = await self._extract_products_async(browser, job_id, site, query, page)
                
                if not products:
                    break
                
                all_products.extend(products)
                
                # Save products to database
                self.db.save_products(products, job_id)
                self.db.update_job_status(job_id, 'running', products_found=len(all_products))
                
                # Random delay
                await asyncio.sleep(self.utils.random_delay(1, 2))
            
            # Complete job
            self.db.update_job_status(
                job_id, 
                'completed', 
                completed_at=datetime.utcnow(),
                total_pages=page,
                products_found=len(all_products)
            )
            
        except Exception as e:
            self.db.update_job_status(
                job_id, 
                'failed', 
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
        finally:
            browser.close()
    
    def _build_url(self, site, query, page):
        """Build search URL"""
        if site.lower() == 'tokopedia':
            return f"https://www.tokopedia.com/search?st=product&q={query}&page={page}"
        return ""
    
    async def _extract_products_async(self, browser, job_id, site, query, page):
        """Extract products asynchronously"""
        products = []
        
        try:
            from selenium.webdriver.common.by import By
            
            # Scroll to load content
            for i in range(3):
                browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1)
            
            # Find product elements
            elements = browser.driver.find_elements(By.CSS_SELECTOR, 'div[class="css-5wh65g"]')
            
            for element in elements[:20]:  # Limit per page
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, 'span[class="+tnoqZhn89+NHUA43BpiJg=="]')
                    price_elem = element.find_element(By.CSS_SELECTOR, 'div[class="urMOIDHH7I0Iy1Dv2oFaNw=="]')
                    link_elem = element.find_element(By.CSS_SELECTOR, 'a')
                    
                    product = {
                        'name': name_elem.text.strip(),
                        'price': price_elem.text.strip(),
                        'url': link_elem.get_attribute('href'),
                        'site': site,
                        'query': query,
                        'page_number': page
                    }
                    
                    if product['name'] and product['name'] != "Name not found":
                        products.append(product)
                        
                except:
                    continue
                    
        except Exception as e:
            print(f"Error extracting products: {e}")
        
        return products
    
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
        """Stop a running scraping job"""
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            
            # Update job status in database
            self.db.update_job_status(
                job_id, 
                'stopped', 
                completed_at=datetime.utcnow(),
                error_message="Job stopped by user"
            )
            
            # Remove from running jobs
            del self.running_jobs[job_id]
            return True
        
        return False
    
    def list_jobs(self):
        """List all scraping jobs"""
        from src.models.database import ScrapingJob
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