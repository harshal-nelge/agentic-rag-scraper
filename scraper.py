import requests
import time
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from fake_useragent import UserAgent

class WebScraper:
    """
    Simple web scraper with BeautifulSoup primary and Selenium fallback.
    """
    
    def __init__(self, delay=1.0, max_retries=3, timeout=30):
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.ua = UserAgent()
        self.driver = None
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_selenium_driver(self):
        """Setup Selenium Chrome driver."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.ua.random}')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.implicitly_wait(10)
            return driver
        except Exception as e:
            self.logger.error(f"Failed to setup Selenium driver: {e}")
            return None
    
    def scrape_with_requests(self, url):
        """Scrape using requests and BeautifulSoup."""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Scraping {url} with requests (attempt {attempt + 1})")
                
                self.session.headers['User-Agent'] = self.ua.random
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                time.sleep(self.delay)
                
                # Check if content is meaningful
                text_content = soup.get_text().strip()
                if len(text_content) > 100:  # Has meaningful content
                    return soup, response.text
                
            except Exception as e:
                self.logger.warning(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                
        return None, None
    
    def scrape_with_selenium(self, url):
        """Scrape using Selenium for JavaScript-heavy sites."""
        try:
            if not self.driver:
                self.driver = self._setup_selenium_driver()
                if not self.driver:
                    return None, None
            
            self.logger.info(f"Scraping {url} with Selenium")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(3)  # Wait for dynamic content
            
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            time.sleep(self.delay)
            return soup, page_source
            
        except Exception as e:
            self.logger.error(f"Selenium failed for {url}: {e}")
            return None, None
    
    def scrape_url(self, url):
        """
        Main scraping method. Returns (soup, raw_html) tuple.
        """
        if not url or not url.startswith(('http://', 'https://')):
            self.logger.error(f"Invalid URL: {url}")
            return None, None
        
        # Try requests first
        soup, raw_html = self.scrape_with_requests(url)
        if soup:
            return soup, raw_html
        
        # Fallback to Selenium
        self.logger.info(f"Falling back to Selenium for {url}")
        return self.scrape_with_selenium(url)
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        
        if self.session:
            self.session.close()
