"""Utility functions for web scraping operations.

This module provides utility functions for creating human-like behavior
during web scraping, including random delays, sleep functions, and
scrolling patterns to avoid detection.

Example:
    Basic usage of utility functions::

        utils = RandomUtils()
        delay = utils.random_delay(1, 3)
        utils.random_sleep(0.5, 2.0)
        utils.human_like_scroll(driver)

Note:
    These utilities help make scraping behavior appear more natural
    and reduce the likelihood of being detected as automated traffic.
"""
import random
import time
from typing import Union
from selenium.webdriver.remote.webdriver import WebDriver

class RandomUtils:
    """Utility functions for random delays and human-like behavior.
    
    This class provides static methods for creating natural, human-like
    behavior patterns during web scraping operations. All methods are
    designed to help avoid bot detection by introducing randomness.
    
    Example:
        >>> utils = RandomUtils()
        >>> delay = utils.random_delay(1.0, 3.0)
        >>> utils.random_sleep(0.5, 2.0)
    
    Note:
        All timing values are optimized for modern websites while
        maintaining effectiveness in avoiding detection.
    """
    
    @staticmethod
    def random_delay(min_delay: float = 0.5, max_delay: float = 3.0) -> float:
        """Generate random delay duration with optimized timing.
        
        Creates a random delay duration within the specified range,
        useful for introducing natural pauses in scraping operations.
        
        Args:
            min_delay (float, optional): Minimum delay in seconds. Defaults to 0.5.
            max_delay (float, optional): Maximum delay in seconds. Defaults to 3.0.
        
        Returns:
            float: Random delay duration in seconds
        
        Example:
            >>> utils = RandomUtils()
            >>> delay = utils.random_delay(1.0, 5.0)
            >>> print(f"Waiting {delay:.2f} seconds")
            >>> time.sleep(delay)
        
        Note:
            Uses uniform distribution for natural randomness.
            Timing is optimized for modern web scraping needs.
        """
        return random.uniform(min_delay, max_delay)
    
    @staticmethod
    def random_sleep(min_sec: float = 0.5, max_sec: float = 2) -> None:
        """Sleep for a random duration with visual feedback.
        
        Pauses execution for a random duration and provides console
        feedback about the sleep duration.
        
        Args:
            min_sec (float, optional): Minimum sleep duration in seconds. Defaults to 0.5.
            max_sec (float, optional): Maximum sleep duration in seconds. Defaults to 2.
        
        Example:
            >>> utils = RandomUtils()
            >>> utils.random_sleep(1.0, 3.0)
            😴 Slept for 2.15 seconds
        
        Note:
            Provides visual feedback for debugging and monitoring.
            Optimized timing for effective bot detection avoidance.
        """
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
        print(f"😴 Slept for {delay:.2f} seconds")
    
    @staticmethod
    def human_like_scroll(driver: WebDriver, scroll_pause_time: float = 1) -> None:
        """Scroll like a human with optimized timing.
        
        Performs gradual scrolling to the bottom of the page, mimicking
        human scrolling behavior with pauses between scroll actions.
        
        Args:
            driver (WebDriver): Selenium WebDriver instance
            scroll_pause_time (float, optional): Pause between scrolls in seconds.
                Defaults to 1.
        
        Example:
            >>> from selenium import webdriver
            >>> driver = webdriver.Chrome()
            >>> driver.get("https://example.com")
            >>> utils = RandomUtils()
            >>> utils.human_like_scroll(driver, 1.5)
        
        Note:
            Scrolls until no new content is loaded, indicating the
            page bottom has been reached. Timing is optimized for
            modern dynamic content loading.
        """
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

