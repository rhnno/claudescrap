"""
SessionLogger: Mengelola State scraping session ke file json,
agar bisa lanjut semisal main.py crash ditengah jalan bisa lanjut lagi
"""
import json
import os
from datetime import datetime
from typing import Optional

class SessionLogger:
    """
    Menyimpan prgress scraping ke disk agar bisa di-resume jika crash.

    Struktur session log di disk:
    {
        "session_id": "{query}_{timestamp}",
        "query": "laptop gaming",
        "status": "in_progress",  # in_progress, completed, failed
        "last_completed_page": 3,       # Halaman terakhir yang SELESAI penuh
        "scraped_products_urls": [],  # List URL produk yang sudah berhasil di-scrape"
        "created_at": "2024-06-01T12:00:00",
        "updated_at": "2024-06-01T12:30:00"
        
    }
    Menggunakan Url sebagai 
    
    """

    # Folder tempat semua session log disimpan
    SESSION_DIR = "data/sessions"

    def __init__(self, query: str) -> None:
        """
        Inisialisasi sesion untuk sebuah query.

        Kalau session untuk query ini sudah ada (artinya sebelumnya crash),
        session lama akan di-load. kalau belum ada, session baru dibuat.

        Args:
            query (str): Query pencarian yang akan di-scrape, misal "laptop gaming"
        """
        # Make sure session dir exists
        os.makedirs(self.SESSION_DIR, exist_ok=True)
    
        self.query = query

        # Make session_id with unique query + timestamp
        # we change space to underscore for safe file naming
        query_slug = query.replace(" ", "_").lower()
        self.session_id = f"{query_slug}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.filepath = os.path.join(self.SESSION_DIR, f"{query_slug}.json")
        # Note : we use query_slug as filename, so if same query is run again,
        # it will overwrite previous session log for that query

        # Load old session if exists, otherwise create new session
        existing = self._load_from_disk()


        if existing and existing.get("status") == "in_progress":
            # Resume old session
            self.data = existing
            print(f"Resuming session : page {self.data['last_completed_page'] + 1}")
        else:
            # Start new session
            self.data = {
                "session_id": self.session_id,
                "query": query,
                "status": "in_progress",
                "last_completed_page": 0,   # 0 its means theres no page completed yet
                "scraped_products_urls": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            self._save_to_disk()
            print(f" Starting new session: {self.session_id}")

    def _load_from_disk(self) -> Optional[dict]:
        """
        Read session log from disk.
        return none if file not exist or error.
        cannot write anything to disk, self.data not necessary initialized yet,
        so just return dict if success, otherwise none.
        """
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, 'r', encoding="utf-8" ) as f:
                return json.load(f)
        
        except json.JSONDecodeError:
            print("session log file is corrupted, starting new session.")
            return None

    def _save_to_disk(self) -> None:
        """
        write self.data to JSON file.
        Can ONLY be called after self.data is initialized,
        so we can be sure data structure is correct.
        """
        self.data["updated_at"] = datetime.now().isoformat()
        with open(self.filepath, 'w', encoding="utf-8" ) as f:
            # indent 2 for pretty print
            json.dump(self.data, f, indent=2, ensure_ascii=False)


    def mark_page_completed(self, page_number: int, product_urls: list) -> None:
        """
        getting called after a page is successfully scraped.
        Args:
            page_number: int, page number that just completed
            product_urls: List URL product that found on t this page
        """
        self.data["last_completed_page"] = page_number

        # add new URL, but make sure no duplicate
        # we use set to ensure uniqueness, then convert back to list
        existing_urls = set(self.data["scraped_products_urls"])
        new_urls = [url for url in product_urls if url not in existing_urls]
        self.data["scraped_products_urls"].extend(new_urls)

        self._save_to_disk()
        print(f"✅ Page {page_number} completed -- new url added: {len(new_urls)}")

    def is_url_scraped(self, url: str) -> bool:
        """
        Check if a product URL has already been scraped in this session.
        Use for deduplication while scraping product details,
        so we dont scrape same product twice even if it appears in multiple pages.
        """
        return url in self.data["scraped_products_urls"]
    
    def get_resume_page(self) -> int:
        """
        rollback to umber 1 page that NOT completed yet.
        If last_completed_page is 3, it means page 1,2,3 are completed,
        so next page to scrape is 4.
        """
        return self.data["last_completed_page"] + 1
    
    def mark_completed(self) -> None:
        """
        Mark session as completed.
        """
        self.data["status"] = "completed"
        self._save_to_disk()
        total = len(self.data["scraped_products_urls"])
        print(f"🎉 Session completed! Total products scraped: {total}")

