Examples
========

This section provides practical examples of using ClaudeScrap for various scraping scenarios.

Basic Scraping Example
----------------------

Simple product scraping from Tokopedia:

.. code-block:: python

   from src.ace import ScrapingOrchestrator
   
   # Initialize orchestrator
   orchestrator = ScrapingOrchestrator()
   orchestrator.setup_browser(headless=True, use_profile=True)
   
   # Configure scraping
   config = {
       "sites": [{
           "name": "tokopedia",
           "queries": ["laptop gaming", "smartphone android"],
           "max_pages": 3,
           "scroll_depth": 3
       }],
       "output_format": ["csv", "excel"],
       "delay_range": [1, 3]
   }
   
   # Run scraping
   try:
       results = orchestrator.run_batch_scraping(config)
       print(f"Scraping completed! Found data for {len(results)} sites")
   finally:
       orchestrator.close()

API Usage Example
-----------------

Using the REST API for remote scraping:

.. code-block:: python

   import asyncio
   from src.services.scraper_service import ScraperService
   
   async def api_scraping_example():
       service = ScraperService()
       
       # Start scraping job
       job_id = await service.start_scraping_job(
           site='tokopedia',
           query='gaming laptop RTX 4060',
           max_pages=5
       )
       
       print(f"Started job: {job_id}")
       
       # Monitor job progress
       while True:
           status = service.get_job_status(job_id)
           print(f"Job {job_id}: {status['status']} - Page {status.get('current_page', 0)}")
           
           if status['status'] in ['completed', 'failed']:
               break
               
           await asyncio.sleep(5)
       
       # Get final results
       final_status = service.get_job_status(job_id)
       print(f"Final results: {final_status.get('products_found', 0)} products found")
   
   # Run the example
   asyncio.run(api_scraping_example())

Browser Management Example
--------------------------

Advanced browser configuration with auto-login:

.. code-block:: python

   from src.utils.browser import BrowserManager
   
   # Initialize browser with custom settings
   browser = BrowserManager(
       use_profile=True,
       profile_name="my_scraping_profile",
       headless=False
   )
   
   try:
       # Setup browser with anti-detection
       driver = browser.setup_driver()
       
       # Auto-login to site (requires credentials in config)
       if browser.auto_login("tokopedia"):
           print("Successfully logged in!")
           
           # Navigate and scrape
           browser.navigate_to("https://www.tokopedia.com/search?q=laptop")
           
           # Your scraping logic here
           
       else:
           print("Login failed, continuing without authentication")
           
   finally:
       browser.close()

Data Processing Example
-----------------------

Processing scraped data with built-in utilities:

.. code-block:: python

   from src.utils.parser import ProductParser
   from src.utils.storage import DataStorage
   
   # Parse HTML files
   parser = ProductParser()
   products = parser.parse_all_files("data/raw_html")
   
   print(f"Parsed {len(products)} products")
   
   # Save to multiple formats
   storage = DataStorage()
   
   # Save to CSV
   csv_path = storage.save_to_csv(products, "gaming_laptops")
   print(f"CSV saved: {csv_path}")
   
   # Process individual products
   for product in products[:5]:  # First 5 products
       print(f"Product: {product['Product Title']}")
       print(f"Price: {product['Price']}")
       print(f"Shop: {product['Shop Name']}")
       print("---")

Database Integration Example
----------------------------

Using the database for persistent storage:

.. code-block:: python

   from src.models.database import DatabaseManager
   
   # Initialize database
   db = DatabaseManager()
   
   # Create a job
   job = db.create_job("my-job-123", "tokopedia", "gaming laptop")
   print(f"Created job: {job.job_id}")
   
   # Update job status
   db.update_job_status("my-job-123", "running", current_page=2, total_pages=5)
   
   # Save products
   products_data = [
       {
           "name": "Gaming Laptop ASUS ROG",
           "price": "Rp 15.000.000",
           "url": "https://tokopedia.com/product/123",
           "site": "tokopedia",
           "query": "gaming laptop"
       }
   ]
   
   db.save_products(products_data, "my-job-123")
   
   # Complete job
   db.update_job_status("my-job-123", "completed", products_found=len(products_data))

Configuration Examples
----------------------

Custom scraping configuration:

.. code-block:: json

   {
       "sites": [
           {
               "name": "tokopedia",
               "queries": [
                   "laptop gaming RTX 4060",
                   "smartphone android flagship",
                   "headphone wireless noise cancelling"
               ],
               "max_pages": 10,
               "scroll_depth": 3
           },
           {
               "name": "shopee",
               "queries": ["laptop murah", "hp android"],
               "max_pages": 5,
               "scroll_depth": 2
           }
       ],
       "output_format": ["csv", "excel", "json"],
       "delay_range": [2, 5]
   }

Error Handling Example
----------------------

Robust error handling for production use:

.. code-block:: python

   from src.ace import ScrapingOrchestrator
   import logging
   
   # Setup logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def robust_scraping(config):
       orchestrator = None
       try:
           orchestrator = ScrapingOrchestrator()
           
           # Setup browser with retry logic
           max_retries = 3
           for attempt in range(max_retries):
               try:
                   if orchestrator.setup_browser(headless=True):
                       logger.info("Browser setup successful")
                       break
               except Exception as e:
                   logger.warning(f"Browser setup attempt {attempt + 1} failed: {e}")
                   if attempt == max_retries - 1:
                       raise Exception("Failed to setup browser after all retries")
           
           # Run scraping with error handling
           results = orchestrator.run_batch_scraping(config)
           
           if results:
               logger.info(f"Scraping completed successfully: {len(results)} sites processed")
               return results
           else:
               logger.warning("No results obtained from scraping")
               return {}
               
       except Exception as e:
           logger.error(f"Scraping failed: {e}")
           return {}
           
       finally:
           if orchestrator:
               try:
                   orchestrator.close()
                   logger.info("Browser closed successfully")
               except Exception as e:
                   logger.error(f"Error closing browser: {e}")
   
   # Usage
   config = {
       "sites": [{"name": "tokopedia", "queries": ["laptop"]}],
       "output_format": ["csv"]
   }
   
   results = robust_scraping(config)