import random
import time

class RandomUtils:
    """Utility functions for random delays and human-like behavior"""
    
    @staticmethod
    def random_delay(min_delay=1, max_delay=10):
        """Generate random delay duration"""
        return random.uniform(min_delay, max_delay)
    
    @staticmethod
    def random_sleep(min_sec=1, max_sec=5):
        """Sleep for random duration"""
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
        print(f"😴 Slept for {delay:.2f} seconds")
    
    @staticmethod
    def human_like_scroll(driver, scroll_pause_time=2):
        """Scroll like a human"""
        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for new content to load
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

