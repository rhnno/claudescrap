import os
import csv

class DataStorage:
    """Handles all file operations"""
    
    def __init__(self):
        self.html_folder = "data/raw_html"
        self.csv_folder = "data/processed"
        self._ensure_folders()
    
    def _ensure_folders(self):
        """Create necessary folders"""
        os.makedirs(self.html_folder, exist_ok=True)
        os.makedirs(self.csv_folder, exist_ok=True)
    
    def save_html(self, html_content, query, page_num):
        """Save HTML content to file"""
        filename = f"page_{query}_{page_num:03d}.html"
        filepath = os.path.join(self.html_folder, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✅ Saved: {filepath}")
        return filepath
    
    def save_to_csv(self, listings, query):
        """Save listings to CSV file"""
        filename = f"Product_Data_{query}.csv"
        filepath = os.path.join(self.csv_folder, filename)
        
        fieldnames = [
            'Product Title', 'Price', 'Sold', 'discount', 
            'Before Discount Price', 'Shop Name', 'location', 
            'Rating', 'Link Product'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(listings)
        
        print(f"✅ CSV saved: {filepath}")
        return filepath

