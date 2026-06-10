"""
ScrapingRunner: He's the one who manage SessionLogger and PageScrapper.

This Class resposibility:
    1. Inisiate session (new or resume if there's a previous session)
    2. Loop page by page until max_page
    3. delegate scraping to PageScraper
    4. Save scraped URLs to SessionLogger

This class is the "orchestrator" of the scraping page process.
He does not know how to scroll, or even parsing an HTML page.
He just managing the flow, and save progress,
if program crashes he does resume.
"""
import traceback
import time
from src.scrapers.page_scraper import PageScraper
from src.utils.browser import BrowserManager
from src.utils.session_logger import SessionLogger

class ScrapingRunner:
    """
    Orchestrate multi-page scrapinmg with resume capability.

    This class is the definition of Manager in a project. 
    """
    def __init__(
            self,
            query:str,
            max_pages: int = 10,
            headless: bool = False
    ) -> None:
        """
        Args:
            query: The search query to scrape, e.g. "Gaming Laptop"
            max_pages: maximum number of pages to scrape.
            headless: True = browser not being opened, False = browser will be seen.        
        """
        self.query = query
        self.max_pages = max_pages
        self.headless = headless

        # inisiate session logger
        # this the one that gonna decide if we start over or resume from the crash point
        self.session = SessionLogger(query)
        
        # browser and scraper not inisiate yet here
        # we make when run() called, so the resource
        # only used when we need to run the scraping process
        self.browser = None
        self.scraper = None

    def run(self):
        """
        running the scraping process, from resume until maximum page.

        Returns:
            dict containing summary of the scraping process, including:
            {
                "query": self.query,
                "pages_scraped": 8,
                "total_urls": 80,
                "status": "completed" / "interrupteed"
            }
        """
        # this decide if we start over or resume from the crash point
        # if new session, start from page 1, else resume from last page
        start_page = self.session.get_resume_page()

        # if session already completed, therefore we don't need to run again. 
        if self.session.data["status"] == "completed":
            print(f" Session '{self.query}' already completed before")
            print(f" Total URL scraped: {len(self.session.data['scraped_product_urls'])}")
            return self._build_summary("completed")
        
        print(f"\n{'='*50}")
        print(f"Query    : {self.query}")
        print(f"Page     : {start_page}/{self.max_pages}")
        print(f"\n{'='*50}")

        # setup browser - this is the one browser getting called
        # if we fail, we can't continue
        if not self._setup_browser():
            return self._build_summary("interrupted")
        
        try: 
            pages_scraped = 0

            for page_num in range(start_page, self.max_pages + 1):
                print(f"\n[Page {page_num}/{self.max_pages}]")

                # setup url for the page
                url = self.scraper.build_search_url(page_num)

                # Delegate scraping to PageScraper
                product_urls = self.scraper.scrape_page(page_num, url)

                if not product_urls:
                    print(f" X No product found on page {page_num}")
                    print(f" -> Stop scraping (propably end of result)")
                    break

                # Note proggress into session log after each page 'success'
                # this order is important: save to disk first, write down later.
                self.session.mark_page_completed(page_num, product_urls)
                pages_scraped += 1

                # pause for every switch page so we don't get blocked by the server
                # only do pause if there's still more page to scrape
                if page_num < self.max_pages:
                    print(f" Pause before going next page...")
                    time.sleep(2)

            self.session.mark_completed()
            return self._build_summary("completed", pages_scraped)

        except KeyboardInterrupt:
            print(f"\n Scraping stopped manually by {KeyboardInterrupt}")
            print(f" Progress saved, resume later any time")
            return self._build_summary("interuppted")
        
        finally:
            # finally always exxecuted, no matter what happen
            # this to make sure browser always closed and RAM cleaned up
            self._teardown_browser()


    def _setup_browser(self) -> bool:
        """
        Inisiate BrowserManager and PageScraper.

        separated from __init__ because we make sure resource browser not created,
        before run() even called.

        Returns:
            True if success, False if failed.
        """ 
        try:
            print(" Setting up browser...")
            self.browser = BrowserManager(
                headless=self.headless,
                use_profile=False # we dont need to use profile, since i turn off the function
                )
            print(" call setup_driver...")
            self.browser.setup_driver()
            
            print(" make PageScraper instance...")
            self.scraper = PageScraper(
                browser=self.browser,
                query=self.query
            )
            return True
        
        except Exception as e:
            print(f" Failed to setup browser: {e}")
            traceback.print_exc()
            return False
        
    def _teardown_browser(self) -> None:
        """
        Close browser and clean up resource.
        Always called in finally block. so we make sure the resource always clean up,
        no mater what fck happen.
        """
        if self.browser:
            print("\n close browser....")
            self.browser.close()
            self.browser = None
            self.scraper = None
            print(" Browser closed, resource cleaned up")
    
    def _build_summary(
            self,
            status: str,
            pages_scraped: int = 0
    ) -> dict:
        """
        Build dictionary summary of scraping result.
        only written if scraping process completed or interrupted
        """
        urls = self.session.data.get("scraped_product_urls", [])
        return {
            "query": self.query,
            "pages_scraped": pages_scraped,
            "total_urls": len(urls),
            "status": status

        }
