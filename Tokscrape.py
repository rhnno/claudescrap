from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
import time
import random
import pandas as pd
import csv
import selenium
print("Selenium:", selenium.__version__)
import re



def scrape_tokopedia_listings(search_query, depth_scroll):
    search_query_url = search_query.replace(' ', '%20')
    
    url = f"https://www.tokopedia.com/search?navsource=home&ob=5&search_id=20250821154911CAF1A3DFB505FF3D4RJW&source=universe&srp_component_id=04.06.00.00&st=product&q={search_query_url}"

    service = Service("drivers/geckodriver.exe")
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")  # Uncomment this line to run in headless mode
    driver = webdriver.Firefox(options=options)
    driver.get(url)

    def is_session_active(driver):
        try:
            driver.current_url
            return True
        except (InvalidSessionIdException, WebDriverException):
            return False

    if not is_session_active(driver):
        print("Session expired, reinitializing driver...")
        driver = webdriver.Firefox(options=options)
        driver.get(url)

    #randomize = random.uniform(1,2)

    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH,'.//*[contains(@class, "css-jza1fo")]'))  #wrapper produk
    )
    time.sleep(2)
    # Simpan debug HTML
    with open("debug_tokped.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    scroll_range = 500
    for i in range(1, depth_scroll + 1):
        end = scroll_range * i
        task = "window.scrollTo(0,"+str(end)+")"
        driver.execute_script(task)
        print("loading to-"+str(i))
        time.sleep(1)
 

    product_items = driver.find_elements(By.XPATH, './/*[contains(@class, "css-5wh65g")]')
    print(f"Jumlah produk ditemukan: {len(product_items)}")

    results = []

    


    for item in product_items[:]:
        try:
            spans = item.find_elements(By.XPATH, './/span')
            divs = item.find_elements(By.XPATH, './/div')
            flip_spans = item.find_elements(By.XPATH, './/span[contains(@class, "fsEljOjqhjSKqjE")]')
            product_link = "N/A"
            
            try:
                # Primary method
                product_link = item.find_element(
                    By.CSS_SELECTOR, 'a[data-theme="default"]'
                ).get_attribute("href")
            except:
                # Fallback method
                anchors = item.find_elements(By.TAG_NAME, "a")
                for a in anchors:
                    href = a.get_attribute("href")
                    if href and "tokopedia.com" in href:
                        product_link = href
                        break

            try:
                product_name = item.find_element(By.XPATH, './/*[contains(@class,"tnoqZhn89")]').text
            except:
                product_name = "N/A"
            
            try:
                product_price = item.find_element(By.XPATH, './/*[contains(@class, "urMOIDHH7")]').text
            except:
                product_price = "N/A"

            try:
                discount_percentage = item.find_element(By.XPATH, './/span[contains(@class, "_7UCYdN8MrOTwg0MKcGu8zg==")]').text
            except:
                # If no discount percentage is found, set it to "0%"
                discount_percentage = "0%"

            try:
                before_discount_price = item.find_element(By.XPATH, './/span[contains(@class, "hC1B8wTAoPszbEZj80w6Qw==")]').text
            except:
                # If no before discount price is found, set it to "N/A"
                before_discount_price = "N/A"
            
            try:        
                product_sell = item.find_element(By.XPATH, './/*[contains(text(), "terjual")]').text
                if product_sell == "":
                        product_sell = "0 sold"
                else:
                        num = product_sell.split()[0]
                        product_sell = f"{num} sold"
            except:
                product_sell = "0 Sold"

            try:
                product_shop = item.find_element(By.XPATH, './/*[contains(@class, "si3CNdiG8AR0EaXvf6bFbQ")]').text
            except:
                product_shop = "N/A"

            try:
                product_rating = item.find_element(By.XPATH, './/*[contains(@class, "55aCJ8bEsyw")]').text
            except:
                product_rating = "N/A"

            try:
                flip_spans = item.find_elements(By.XPATH, './/span[contains(@class, "flip")]')
                # take all text values, remove empty ones
                location_texts = [span.text.strip() for span in flip_spans if span.text.strip()]
                product_location = " | ".join(location_texts) if location_texts else "N/A"
            except:
                product_location = "N/A"
            




            product_info = {
                'Product Name': product_name,
                'Price': product_price,
                'discount': discount_percentage,
                'Before Discount Price': before_discount_price,
                'Sold': product_sell,
                'Shop Name' : product_shop,
                'Rating': product_rating,
                'Link Product': product_link
            }

            results.append(product_info)
        except Exception as e:
            print(f"Gagal ambil data produk: {e}")
            continue

    driver.quit()
    return results


if __name__ == '__main__':
    search_query = 'Kopi Bubuk'
    depth_scroll = 5  # Adjust the depth of scrolling as needed
    listings = scrape_tokopedia_listings(search_query, depth_scroll)

    
    def filename_csv(query):
        safename = re.sub(r'[\\/*?:"<>|]', '', query)
        safename = safename.replace(' ', '_')
        return f"{safename}.csv"

    def filename_excel(query):
        safename = re.sub(r'[\\/*?:"<>|]', '', query)
        safename = safename.replace(' ', '_')
        return f"{safename}.xlsx"


    csv_filename = filename_csv(search_query)
    excel_filename = filename_excel(search_query)

    if listings:
        fieldnames = ['Product Name', 'Price', 'Sold','discount','Before Discount Price', 'Shop Name','location', 'Rating', 'Link Product']
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(listings)

    if listings:
        df = pd.DataFrame(listings)
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Tokopedia Listings')

    print("Data berhasil disimpan ke '{csv_filename}' dan '{excel_filename}'.")

    for index, listing in enumerate(listings, start=1):
        print(f"Product {index}:")
        print(f"Name: {listing['Product Name']}")
        print(f"Price: {listing['Price']}")
        print(f"Terjual: {listing['Sold']}")
        print(f"Toko: {listing['Shop Name']}")
        print(f"Rating: {listing['Rating']}")
        print(f"Link: {listing['Link Product']}")
        print("=" * 30)
