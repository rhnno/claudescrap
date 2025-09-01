from bs4 import BeautifulSoup
import html
import os

class ProductParser:
    """Parses HTML content to extract product data"""
    
    def __init__(self):
        self.selectors = {
            'product_cards': 'div.css-5wh65g',
            'title': 'div.css-1f4mp12',
            'price': 'div.css-rhd610', 
            'discount': 'span._7UCYdN8MrOTwg0MKcGu8zg==',
            'link': 'a[href]',
            'location': 'span.gxi+fsEljOjqhjSKqjE+sw==.flip',
            'shop': 'span.si3CNdiG8AR0EaXvf6bFbQ==',
            'sold': 'span.u6SfjDD2WiBlNW7zHmzRhQ==',
            'rating': 'span._2NfJxPu4JC-55aCJ8bEsyw==',
            'original_price': 'span.hC1B8wTAoPszbEZj80w6Qw=='
        }
    
    def parse_single_file(self, filepath):
        """Parse a single HTML file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            return self._extract_products(soup)
        
        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
            return []
    
    def parse_all_files(self, folder="data/raw_html"):
        """Parse all HTML files in folder"""
        all_listings = []
        
        if not os.path.exists(folder):
            print(f"❌ Folder {folder} doesn't exist")
            return all_listings
        
        html_files = [f for f in os.listdir(folder) if f.endswith('.html')]
        
        for filename in html_files:
            filepath = os.path.join(folder, filename)
            print(f"📝 Parsing: {filename}")
            
            listings = self.parse_single_file(filepath)
            all_listings.extend(listings)
        
        print(f"✅ Total products found: {len(all_listings)}")
        return all_listings
    
    def _extract_products(self, soup):
        """Extract product data from BeautifulSoup object"""
        listings = []
        product_cards = soup.find_all('div', class_='css-5wh65g')
        
        for card in product_cards:
            try:
                product_data = self._extract_single_product(card)
                if product_data:
                    listings.append(product_data)
            except Exception as e:
                print(f"❌ Error extracting product: {e}")
                continue
        
        return listings
    
    def _extract_single_product(self, card):
        """Extract data from a single product card"""
        def safe_extract(selector_key, attr='text'):
            try:
                element = card.find(self.selectors[selector_key].split('.')[0], 
                                  class_='.'.join(self.selectors[selector_key].split('.')[1:]))
                if element:
                    if attr == 'href':
                        return element.get('href', 'N/A')
                    else:
                        return html.unescape(element.get_text(strip=True))
                return 'N/A'
            except:
                return 'N/A'
        
        return {
            'Product Title': safe_extract('title'),
            'Price': safe_extract('price'),
            'Sold': safe_extract('sold'),
            'discount': safe_extract('discount'),
            'Before Discount Price': safe_extract('original_price'),
            'Shop Name': safe_extract('shop'),
            'location': safe_extract('location'),
            'Rating': safe_extract('rating'),
            'Link Product': safe_extract('link', 'href')
        }