import time
from selenium.webdriver.common.by import By

class BaseScraper:
    def __init__(self, browser):
        self.browser = browser

    def scrape(self, url):
        """
        A simple scraper that navigates to a URL and extracts some data.
        """
        self.browser.navigate_to(url)
        time.sleep(3)

        # This is just an example. The actual selectors would be more complex.
        products = []
        product_elements = self.browser.driver.find_elements(By.CSS_SELECTOR, ".product-item")

        for element in product_elements:
            products.append({
                "name": element.find_element(By.CSS_SELECTOR, ".product-name").text,
                "price": element.find_element(By.CSS_SELECTOR, ".product-price").text,
            })

        return products
