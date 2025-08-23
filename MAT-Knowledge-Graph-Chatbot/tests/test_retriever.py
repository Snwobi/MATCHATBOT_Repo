"""
Tests for document retriever
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from retriever import DocumentRetriever

class TestDocumentRetriever:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_documents = pd.DataFrame({
            'text': [
                'MAT standards are important for treatment',
                'Medication assisted treatment helps patients',
                'Public health scotland provides guidance',
                'Recovery is the main goal of MAT'
            ],
            'source_url': ['url1', 'url2', 'url3', 'url4']
        })
        
        self.retriever = DocumentRetriever(self.test_documents)
        self.retriever.fit()
    
    def test_retriever_initialization(self):
        """Test retriever initializes correctly"""
        assert self.retriever.documents is not None
        assert self.retriever.vectorizer is not None
    
    def test_retrieve_documents(self):
        """Test document retrieval"""
        results = self.retriever.retrieve("MAT standards", k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all('similarity' in result for result in results)
        assert all('text' in result for result in results)
    
    def test_retrieve_by_keywords(self):
        """Test keyword-based retrieval"""
        results = self.retriever.retrieve_by_keywords(['MAT', 'treatment'])
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_get_statistics(self):
        """Test document statistics"""
        stats = self.retriever.get_document_statistics()
        
        assert 'total_documents' in stats
        assert stats['total_documents'] == 4
        assert 'average_length' in stats