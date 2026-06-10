"""
PageScraper : Scrapes HTML pages 1 by 1 from the query results Tokopedia.

Responsible for:
    1. Navigate to page URL
    2. Scroll until all product card loaded(lazy loading)
    3. Save HTML Page to disk
    4. return list product URLs found on this page,
    so we can scrape them later 

It not do parsing product details, just save the whole page as it was,
and we will parse it later in ProductScraper.

It not touch any database.

its just responsible for raw data collection,
and save it to disk for later use.
"""
import os
import time
import json
import hashlib
from typing import Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.utils.browser import BrowserManager
from src.scrapers.stream_collector import StreamCollector


class PageScraper:
    """
    Grab and save hmtl 1 page of an query.

    This class receive BrowserManager that already setuped from browser.py.
    So we dont need to make any shit here again. we use dependency injection for this
    in this code we dont give a fuck bout how the browser got setup,
    all we do just scrape html page.
    """

    # Selector for product card Tokopedia grabbed yaml toml template
    PRODUCT_CARD_SELECTOR = 'div[class="css-5wh65g"]'
    PRODUCT_CARD_SELECTOR2 = 'div[class="gG1uA844gIiB2+C3QWiaKA=="]'
    PRODUCT_LINK_SELECTOR = 'a[href*="tokopedia.com"]'

    # Root folder for every HTML file we save
    RAW_HTML_DIR = "data/raw_html"

    def __init__(self, browser: BrowserManager, query: str) -> None:
        """
        Args:
            browser: pre-setup BrowserManager with driver on it.
            query: The search query for which to scrape results,
            and will be used for subfolder name
        """
        self.browser = browser
        self.query = query

        # Make slug from query for subfolder: we just change 'space' into '_'.
        query_slug = query.replace(" ", "_")
        self.output_dir = os.path.join(self.RAW_HTML_DIR,query_slug)
        os.makedirs(self.output_dir, exist_ok=True)

    def scrape_page(self, page_number: int, url: str) -> list[str]:
        """ Navigate to page and delegate stream process to StreamCollector."""
        print(f" > Navigate into page {page_number}...")

        if not self.browser.navigate_to(url):
            print(f" /x/ Failed to navigate")
            return []
        
        # Wait first card to appear before start streaming
        if not self._wait_for_products():
            print(f" /x/ There's no card appear - because anti-bot or empty page")
            return []

        collector = StreamCollector(
            driver=self.browser.driver,
            query=self.query,
            page_number=page_number
        )
        summary = collector.collect()

        # Return seen_ids as URLs for session log
        # Reconstruct from broze files that written
        return self._load_urls_from_bronze(page_number)
    
    def _load_urls_from_bronze(self, page_number: int) -> list[str]:
        """Read URLs from bronze batch files that has been written."""
        query_slug = self.query.replace(" ", "_").lower()
        bronze_dir = os.path.join("data", "bronze", query_slug)

        urls = []
        if not os.path.exists(bronze_dir):
            return urls
        
        # Take all batch files for current page
        prefix = f"page_{page_number:03d}_"
        for filename in sorted(os.listdir(bronze_dir)):
            if filename.startwith(prefix):
                filepath = os.path.join(bronze_dir, filename)
                with open(filepath) as f:
                    batch = json.load(f)
                    urls.extend([card["url"] for card in batch])
        
        return urls


    def _wait_for_products(self, timeout: int = 15) -> bool:
        """
        just wait until product card appear on DOM.
        
        Return True if card found before timeot,
        Otherwise return False if timeout reached and still no card found.
        """

        try:
            WebDriverWait(self.browser.driver, timeout).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, self.PRODUCT_CARD_SELECTOR2)
                )
            )
            return True
        except Exception:
            return False
    
    def _scroll_page(self, scroll_count: int = 5) -> None:
        """
        Scroll Page down to trigger lazy loading.
        
        we scroll slowly,
        because Tokopedia load incrementally while viewport reach the area card,

        """
        if not self.browser.driver:
            return
        
        total_height = self.browser.driver.execute_script(
            "return document.body.scrollHeight"
        )

        # we add guard - if execute_script failed and return 0,
        #  we use default height rather than scrash with TypeError
        if not total_height or not isinstance(total_height, (int, float)):
            print(" ⚠ Warning: execute_script for scrollHeight failed, using default height 3000")
            self.browser.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(2)
            return

        for i in range(1, scroll_count + 1):
            scroll_to = int(total_height * (i / scroll_count))
            self.browser.driver.execute_script(
                f"window.scrollTo(0, {scroll_to});"
            )
            # a bit delay to let the page load new content after scroll
            time.sleep(1.2)

        # Scroll back to top after done, just for good measure
        self.browser.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    def _extract_product_urls(self) -> list[str]:
        """
        Ekstrak semua URL produk dari DOM halaman saat ini.
        
        Kita ekstrak URL dari DOM live (bukan dari HTML string) karena
        lebih mudah dan lebih akurat — tidak perlu parsing HTML manual.
        URL ini yang akan dicatat di session log untuk deduplikasi.
        """
        if not self.browser.driver:
            return []

        urls = []
        try:
            # Ambil semua link di dalam product card
            cards = self.browser.driver.find_elements(
                By.CSS_SELECTOR, self.PRODUCT_CARD_SELECTOR
            )
            for card in cards:
                try:
                    link = card.find_element(By.CSS_SELECTOR, "a[href]")
                    href = link.get_attribute("href")
                    # Filter hanya URL Tokopedia yang valid (bukan iklan, dll)
                    if href and "tokopedia.com" in href:
                        # Buang query parameter (?src=...) untuk URL yang bersih
                        clean_url = href.split("?")[0]
                        if clean_url not in urls:  # hindari duplikat per halaman
                            urls.append(clean_url)
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠ Error saat ekstrak URL: {e}")

        return urls

    def _save_page_html(self, page_number: int, url: str) -> Optional[str]:
        """
        Simpan HTML halaman saat ini ke file.
        
        Nama file: {query}_{page_number}_{url_hash}.html
        URL hash dipakai untuk memastikan nama file unik meskipun
        ada dua halaman dengan nomor yang sama (misalnya saat retry).
        
        Returns:
            Path file yang disimpan, atau None jika gagal.
        """
        if not self.browser.driver:
            return None

        try:
            html_content = self.browser.driver.page_source

            # Buat hash pendek dari URL untuk uniqueness
            url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:6]
            query_slug = self.query.replace(" ", "_").lower()
            filename = f"{query_slug}_page_{page_number:03d}_{url_hash}.html"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            return filepath

        except Exception as e:
            print(f"  ✗ Gagal menyimpan HTML: {e}")
            return None

    def build_search_url(self, page_number: int) -> str:
        """
        Bangun URL pencarian Tokopedia untuk halaman tertentu.
        
        Dipisahkan sebagai method tersendiri supaya mudah diganti
        kalau format URL Tokopedia berubah di masa depan.
        """
        query_encoded = self.query.replace(" ", "%20")
        return (
            f"https://www.tokopedia.com/search"
            f"?st=product&q={query_encoded}&page={page_number}"
        )