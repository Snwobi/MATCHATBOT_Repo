"""
Tests for knowledge graph functionality
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from knowledge_graph import KnowledgeGraph

class TestKnowledgeGraph:
    
    @patch('neo4j.GraphDatabase.driver')
    def setup_method(self, mock_driver):
        """Setup test fixtures"""
        self.mock_driver = Mock()
        mock_driver.return_value = self.mock_driver
        self.kg = KnowledgeGraph()
    
    def test_kg_initialization(self):
        """Test KG initializes correctly"""
        assert self.kg.driver is not None
    
    def test_extract_entities(self):
        """Test entity extraction"""
        test_data = pd.DataFrame({
            'text': [
                'MAT Standard 1 is important',
                'Public Health Scotland supports MAT',
                'Treatment in Scotland is improving'
            ]
        })
        
        entities = self.kg._extract_entities(test_data)
        
        assert 'MATStandard' in entities
        assert 'Organization' in entities
        assert 'Location' in entities
        assert len(entities['MATStandard']) > 0
    
    def test_extract_relationships(self):
        """Test relationship extraction"""
        test_data = pd.DataFrame({
            'text': [
                'MAT supports recovery',
                'Standards are implemented by organizations'
            ]
        })
        
        relationships = self.kg._extract_relationships(test_data)
        
        assert 'SUPPORTS' in relationships
        assert 'IMPLEMENTS' in relationships