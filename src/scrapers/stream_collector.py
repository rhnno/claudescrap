"""
StreamCollector: Delta-Based streaming observer for product cards.

The function:
    - Observe DOM delta (new cards that found since lastest scan)
    - Store batch into disk (bronze layer) every new card
    - Handle continuation (scroll, load more, or stop)

He dont know how to navigate URL or even session management.
He just know: "observe, emit, and repeat until stream dead.".
"""
import json
import os
import time
import random
from datetime import datetime
from typing import Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException

class StreamCollector:
    """
    Delta-based DOM observer that stream product cards into bronze layer.

    Main concept:
    - seen_ids: source of truth for 'deduplication'
    - observe_dom_delta(): just return card that 'haven't' been seen.
    - emit_batch(): write 'new card' into JSON on disk
    - stream_alive: flag controlled from result of 'attempt_continuation()'
    
    State that must been protected when streaming:
    {
        "seen_ids" : set(),         # unique product IDs that been processed
        "stagnant_rounds": 0,       # how many stagnant state that haven't appear new card
        "batches_saved": 0,         # how much batch have been writen in disk
        "continuation_attempts": 0, # how many attempt to try hit "Muat Lebih banyak"
        "last_card_count": 0        # total cards in lastest DOM scanned. 
    }
    
    """

    PRODUCT_CARD_SELECTOR = 'div[class="css-5wh65g"]'
    LOAD_MORE_SELECTOR = 'button[data-unify="Button"].css-1turmok-unf-btn'
    # how many stagnant round allowed before trying continuation
    MAX_STAGNANT_ROUNDS = 3
    # how many attempt continuation before considering done
    MAX_CONTINUATION_ATTEMPTS = 9e12
    # MAX card each session for safety
    MAX_CARDS = 10000

    def __init__(
            self,
            driver: WebDriver,
            query: str,
            page_number: int,
            bronze_dir: str= "data/bronze"
    ) -> None:
        """
        Args:
            driver: Webdriver that have been navigated into target page
            query: search keyword (for the foldername as well)
            page_number: Page Number ( for the name batch file)
            bronze_dir: Root folder for bronze layer output
        """
        self.driver = driver
        self.query = query
        self.page_number = page_number

        #setup output dir
        query_slug = query.replace(" ", "_").lower()
        self.output_dir = os.path.join(bronze_dir, query_slug)
        os.makedirs(self.output_dir, exist_ok=True)

        # Internal state - source of truth
        self.state = {
            "seen_ids": set(),
            "stagnant_rounds": 0,
            "batches_saved": 0,
            "continuation_attempts": 0,
            "last_card_count": 0
        }

    def collect(self) -> dict:
        """
        Entry point: running scraping loop until the end
        
        returns:
            Summary dict from streaming should be lookk like this:
            {
            "total_cards": int,
            "batches_saved": int,
            "continuation": int
            }
        """
        print(f" StreamCollector starting streaming page : {self.page_number}...")
        stream_alive = True

        while stream_alive:
            # just observe new/unique cards that haven't been seen
            new_cards = self._observe_dom_delta()

            if new_cards:
                # New card appear and emit to disk then reset the stagnant counter
                self._emit_batch(new_cards)
                self.state["stagnant_rounds"] = 0
                print(f"[Stream] +{len(new_cards)} new card"
                      f"(total seen: {len(self.state['seen_ids'])})")
                
            else:
                # if theres no card, increment stagnant counter
                self.state["stagnant_rounds"] += 1
                print(f" [Stream] Stagnant "
                      f"{self.state['stagnant_rounds']}/{self.MAX_STAGNANT_ROUNDS}")
                
                # Bounce scroll: go up and down to trigger new card
                self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight * 0.7);")
                time.sleep(0.5)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1.3, 2.5))
                
                if self.state["stagnant_rounds"] >= self.MAX_STAGNANT_ROUNDS:
                    # try continuation before stopping
                    stream_alive = self._attempt_continuation()

            
            # safety check so we not loading too much n crashed
            if len(self.state["seen_ids"]) >= self.MAX_CARDS:
                print(f"[Stream] Maximum line {self.MAX_CARDS} card achieved")
                stream_alive = False

            # Scroll to lastest card (PROPABLY THIS GOTTA NEED TO BE CHANGED)
            # CAUSE I DONT REALLY LIKE IT
            if stream_alive:
                self._scroll_to_last_card()
                time.sleep(random.uniform(1.2, 2.1))

        Summary = {
            "total_cards": len(self.state["seen_ids"]),
            "batches_saved": self.state["batches_saved"],
            "continuation_attempts": self.state["continuation_attempts"]
        }
        print(f"[StreamCollector] done - {Summary}")
        return Summary
    
    def _observe_dom_delta(self) -> list[dict]:
        """
        Scan DOM and only return unique card based on seen_ids.

        This ini core of delta-based approach:
        - Not re-process card that have been seen(simply, duplicate)
        - Not save WebElement (load data directly, remove element)
        - Fingerprint based on URL slug - stable and unique

        Returns:
            List dict of new cards: [{
            "id": str,
            "urls": str,
            "raw_cards": str,
            "scraped_at": str
            }]
        """

        new_cards = []

        try:
            # find_elements() - we iterate directly, not saving the list
            for card in self.driver.find_elements(
                By.CSS_SELECTOR, self.PRODUCT_CARD_SELECTOR
            ):
                try:
                    link = card.find_element(By.CSS_SELECTOR, "a[href]")
                    href = link.get_attribute("href")
                    outerhtml = card.find_element(By.CSS_SELECTOR, "a[href]").get_attribute('outerHTML')

                    if not href or "tokopedia.com" not in href:
                        continue
                    # make fingerprint ID from URL slug
                    product_id = self._extract_id(href)

                    # skip if been seen or not unique
                    if product_id in self.state["seen_ids"]:
                        continue

                    # new card found - store into result
                    new_cards.append({
                        "id": product_id,
                        "url": href.split("?"), # clean URL without query params
                        "raw_cards": outerhtml,
                        "scraped_at": datetime.now().isoformat()
                    })

                    # Mark as seen - WebElement can be GC(cleared)
                    self.state["seen_ids"].add(product_id)
                
                except Exception:
                    # skip card that failed extracted - dont stop every loop
                    continue
        
        except Exception as e:
            print(f" [Stream] Error scan DOM: {e}")

        # update last_card_count for monitoring purposes
        self.state["last_card_count"] = len(self.state["seen_ids"])
        return new_cards

    def _emit_batch(self, cards: list[dict]) -> None:
        """
        Write new card to file JSON in bronze layer.

        Naming convention:
        page_001_batch001.json, page_001_batch_002.json, dst

        Every file is array JSON itself standalone no dependency each file. 
        """
        self.state["batches_saved"] += 1
        batch_num = self.state["batches_saved"]

        filename = (
            f"page_{self.page_number:03d}_"
            f"batch_{batch_num:03d}.json"
        )
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)
        
        print(f" [Emit] {filename} ({len(cards)} cards)")
    
    def _attempt_continuation(self) -> bool:
        """
        Try to continue stream when facing stagnant state.

        retry sequence:
        1. Click the "Muat Lebih Banyak" button if it is present.
        2. If the button is not found, the stream has been fully exhausted.
        
        Returns:
            True means succesfully continue (stream keep alive)
            False means theres nothing to continue of fail (stream is dead)
        """
        self.state["continuation_attempts"] += 1
        self.state["stagnant_rounds"] = 0  # reset counter

        if self.state["continuation_attempts"] > self.MAX_CONTINUATION_ATTEMPTS:
            print(f"    [Stream] Max continuation attempts achieved - stream done")
            return False
        
        print(f"    [Stream] Continuation retry"
              f"(attempt {self.state['continuation_attempts']}/"
              f"{self.MAX_CONTINUATION_ATTEMPTS})...")
        
        # Retry click "Muat Lebih Banyak"
        if self._click_load_more():
            print(f"    [Stream] 'Muat Lebih Banyak' click and waiting to load...")
            time.sleep(random.uniform(2.1, 3.9))
            return True     # strean still alive and waiting new card
        
        # There's no button appear, then its considered as done/exhasuted
        print(f"    [Stream] Theres no more continuation - flagged as done")
        return False
    
    def _click_load_more(self) -> bool:
        """ 
        
        click button 'Muat Lebih Banyak' if appear.
        Small Function. 
        """
        try:
            buttons = self.driver.find_elements(
                By.CSS_SELECTOR, self.LOAD_MORE_SELECTOR
            )
            print(f"    [Debug] Tombol ditemukan: {len(buttons)}")

            if not buttons:
                 # try to find with text
                 all_buttons = self.driver.find_elements(By.TAG_NAME, 'button')
                 muat_button = [b for b in all_buttons
                                if 'Muat Lebih Banyak' in b.text]
                 print(f"   [Debug] button 'Muat Lebih Banyak': {len(muat_button)}")
                 for b in muat_button:
                     print(f"   [Debug] Teks: '{b.text}', class: '{b.get_attribute('class')}'")

            button = self.driver.find_element(
                By.CSS_SELECTOR, self.LOAD_MORE_SELECTOR
            )
            # Scroll to 'button'
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                button
            )

            time.sleep(0.5)
            # Click via JS - more readable than .click()
            self.driver.execute_script("arguments[0].click();", button)
            return True
        except NoSuchElementException:
            return False
        except Exception as e:
            print(f"    [Stream] Error to click load more: {e}")
            return False
        
    def _scroll_to_last_card(self) -> None:
        """Scroll to lastest card to trigger lazy load. Small Function"""
        try:
            cards = self.driver.find_elements(
                By.CSS_SELECTOR, self.PRODUCT_CARD_SELECTOR
            )
            if cards:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView("
                    "{behavior: 'smooth', block: 'end'});",
                    cards[-1]
                )
        except Exception:
            #Fallback to scroll into bottom
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
    
    def _extract_id(self, url: str) -> str:
        """
        Extract product ID from URL as unique fingerprint.

        "https://www.tokopedia.com/toko-abc/produk-xyz?src=search"
        -> "toko-abc/produk-xyz"
        lalu dilakukan hash menjadi
        -> hash binary md5 yang diubah jadi hexadecimal 8 karakter
        """
        from urllib.parse import urlparse
        import hashlib
        try:
            # url parse will identify any domain 
            parsed_url = urlparse(url)
            # .path will immediatly grab "/toko-abc/produk-xyz" after /
            clean = parsed_url.path.strip("/")
            url_hash = hashlib.md5(clean.encode(), usedforsecurity=False).hexdigest()[:8]
            return url_hash
        except Exception:
            # Fallback to raw url if parsing failed
            return url