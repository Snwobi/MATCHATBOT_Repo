"""
Tests for data scraper functionality
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_scraper import MATScraper

class TestMATScraper:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.scraper = MATScraper()
    
    def test_scraper_initialization(self):
        """Test scraper initializes correctly"""
        assert self.scraper.urls is not None
        assert len(self.scraper.urls) > 0
        assert self.scraper.session is not None
    
    @patch('requests.Session.get')
    def test_scrape_url_success(self, mock_get):
        """Test successful URL scraping"""
        # Mock response
        mock_response = Mock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.scraper.scrape_url("http://test.com")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Test content" in result[0]
    
    @patch('requests.Session.get')
    def test_scrape_url_failure(self, mock_get):
        """Test URL scraping failure handling"""
        mock_get.side_effect = Exception("Network error")
        
        result = self.scraper.scrape_url("http://test.com")
        
        assert result == []
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = "  This is   a test!@#$   "
        clean_text = self.scraper._clean_text(dirty_text)
        
        assert clean_text == "This is a test!"
    
    def test_clean_data(self):
        """Test data cleaning process"""
        # Create test data
        test_data = pd.DataFrame({
            'text': ['Valid text', '', 'Another valid text', 'Valid text'],
            'source_url': ['url1', 'url2', 'url3', 'url4'],
            'length': [10, 0, 18, 10]
        })
        
        cleaned = self.scraper.clean_data(test_data)
        
        # Should remove empty and duplicate entries
        assert len(cleaned) == 2
        assert '' not in cleaned['text'].values