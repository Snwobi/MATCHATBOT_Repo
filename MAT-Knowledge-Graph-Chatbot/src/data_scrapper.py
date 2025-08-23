"""
Web scraping functionality for MAT-related content
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import List, Dict
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)

class MATScraper:
    """Web scraper for MAT (Medication-Assisted Treatment) content"""
    
    def __init__(self):
        self.urls = config.URLS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("MATScraper initialized")
    
    def scrape_url(self, url: str) -> List[str]:
        """Scrape text content from a single URL"""
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraph tags
            paragraphs = soup.find_all('p')
            texts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            
            logger.info(f"Extracted {len(texts)} paragraphs from {url}")
            return texts
            
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}")
            return []
    
    def scrape_all_urls(self) -> pd.DataFrame:
        """Scrape all configured URLs and return as DataFrame"""
        all_data = []
        
        for url in self.urls:
            texts = self.scrape_url(url)
            for text in texts:
                all_data.append({
                    'source_url': url,
                    'text': text,
                    'length': len(text)
                })
            
            # Be respectful to the server
            time.sleep(1)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Scraped {len(df)} total text segments")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess scraped data"""
        logger.info("Starting data cleaning process")
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0].copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {initial_count - len(df)} duplicate entries")
        
        # Clean text
        df['text'] = df['text'].apply(self._clean_text)
        
        # Filter out very short texts (less than 20 characters)
        df = df[df['text'].str.len() >= 20].copy()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Cleaning complete. Final dataset: {len(df)} entries")
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text entry"""
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\\w\\s.,;:!?()-]', '', text)
        
        return text.strip()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV file"""
        if filename is None:
            filename = config.CLEANED_DATA_FILE
        
        filepath = f"{config.PROCESSED_DATA_PATH}/{filename}"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath