from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains as action
import time
import os
import logging

def setup_driver():
    """Setup Chrome driver with anti-detection options"""
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def save_page_source(driver, page_num, folder="debug_pages"):
    """Save current page source to HTML file"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{folder}/debug_{search_query}_{page_num:03d}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print(f"✅ Saved: {filename}")
    return filename

def wait_for_page_load(driver, timeout=10):
    """Wait for page to fully load"""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        # Additional wait for dynamic content
        time.sleep(2)
        return True
    except TimeoutException:
        print("⚠️  Page load timeout")
        return False

def scroll_to_pagination(driver):
    """Scroll to pagination area"""
    try:
        # Try multiple pagination selectors
        pagination_selectors = [
            #'.//*[contains(@class, "css-5p3bh2-unf-pagination-item")]',
            #'.//nav[contains(@class, "pagination")]',
            #'.//*[contains(@class, "pagination")]',
            #'.//*[contains(text(), "Next")]',
            #'.//*[contains(text(), "›")]',
            './/button[contains(@class, "css-1turmok-unf-btn eg8apji0")]',
            './/svg[contains(@class, "unf-icon)]'
        ]

        pagination_element = None
        for selector in pagination_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    pagination_element = elements[0]
                    break
            except:

                continue
        
        if pagination_element:
            driver.execute_script("arguments[0].scrollIntoView(true);", pagination_element)
            time.sleep(2)
            return pagination_element
        else:
            # Scroll to bottom as fallback
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            return None
    except Exception as e:
        print(f"❌ Scroll error: {e}")
        return None

def find_next_button(driver):
    """Find next page button using multiple strategies"""
    next_button_selectors = [
        # Tokopedia specific
        './/button[contains(@class, "css-1turmok-unf-btn eg8apji0")]'
        
        # Generic selectors
        './/button[contains(text(), "Next")]',
        './/a[contains(text(), "Next")]',
        './/button[contains(text(), "›")]',
        './/a[contains(text(), "›")]',
        './/button[@aria-label="Next"]',
        './/a[@aria-label="Next"]',
        './/svg[contains(@class, "unf-icon)]',
        
        # CSS selectors converted to XPath
        './/*[contains(@class, "pagination-next")]',
        './/*[contains(@class, "next-page")]'
    ]
    
    for selector in next_button_selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    return element
        except:
            continue
    
    return None

def scrape_continuous_pages(start_url, max_pages=100, delay=3):
    """
    Scrape multiple pages continuously and save HTML sources
    
    Args:
        start_url: Starting URL
        max_pages: Maximum number of pages to scrape
        delay: Delay between pages (seconds)
    """
    driver = setup_driver()
    scraped_pages = []
    
    try:
        print(f"🚀 Starting continuous scraping for {max_pages} pages...")
        print(f"📍 Starting URL: {start_url}")
        
        # Go to first page
        driver.get(start_url)
        wait_for_page_load(driver)
        
        for page_num in range(1, max_pages + 1):
            print(f"\n📄 Processing Page {page_num}")
            
            # Wait for page to load completely
            if not wait_for_page_load(driver):
                print(f"❌ Page {page_num} failed to load properly")
                break
            
            # Save current page source
            filename = save_page_source(driver, page_num)
            scraped_pages.append({
                'page': page_num,
                'url': driver.current_url,
                'filename': filename
            })
            
            # Check if we're on the last page we want
            if page_num >= max_pages:
                print(f"✅ Reached maximum pages ({max_pages})")
                break
            
            # Find and scroll to pagination
            print("🔍 Looking for next page button...")
            scroll_to_pagination(driver)
            
            # Find next button
            next_button = find_next_button(driver)
            
            if next_button:
                try:
                    # Try clicking the next button
                    print("▶️  Clicking next page...")
                    driver.execute_script("arguments[0].click();", next_button)
                    
                    # Wait a bit for navigation
                    time.sleep(delay)
                    
                    # Check if URL changed (successful navigation)
                    current_url = driver.current_url
                    if len(scraped_pages) > 1 and current_url == scraped_pages[-2]['url']:
                        print("⚠️  URL didn't change, might be last page")
                        break
                        
                except Exception as e:
                    print(f"❌ Failed to click next button: {e}")
                    break
            else:
                print("❌ Next button not found - reached last page or pagination issue")
                break
        
        print(f"\n🎉 Scraping completed! Total pages scraped: {len(scraped_pages)}")
        
        # Print summary
        print("\n📊 Summary:")
        for page_info in scraped_pages:
            print(f"Page {page_info['page']}: {page_info['filename']}")
            
        return scraped_pages
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        return scraped_pages
    
    finally:
        print("\n🔄 Closing browser...")
        driver.quit()

# Example usage
if __name__ == "__main__":
    # search query
    search_query = "laptop"
    # Replace with your Tokopedia URL
    tokopedia_url = f"https://www.tokopedia.com/search?navsource=home&ob=5&search_id=20250821154911CAF1A3DFB505FF3D4RJW&source=universe&srp_component_id=04.06.00.00&st=product&q={search_query.replace(' ', '%20')}"
    
    # Scrape up to 10 pages (change as needed)
    results = scrape_continuous_pages(
        start_url=tokopedia_url,
        max_pages=10,
        delay=3  # 3 second delay between pages
    )
    #logging
    logger = logging.getLogger(__name__)
    print(f"log: {logger}")
    
    print(f"\n✨ All done! Scraped {len(results)} pages")

