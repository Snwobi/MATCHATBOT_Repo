"""
Document retrieval system for RAG implementation
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)

class DocumentRetriever:
    """Simple TF-IDF based document retriever"""
    
    def __init__(self, documents: pd.DataFrame = None):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.is_fitted = False
        logger.info("DocumentRetriever initialized")
    
    def load_documents(self, filepath: str = None):
        """Load documents from CSV file"""
        if filepath is None:
            filepath = f"{config.PROCESSED_DATA_PATH}/{config.CLEANED_DATA_FILE}"
        
        try:
            self.documents = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.documents)} documents from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise
    
    def fit(self):
        """Fit the TF-IDF vectorizer on the documents"""
        if self.documents is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info("Fitting TF-IDF vectorizer")
        texts = self.documents['text'].fillna('').tolist()
        self.document_vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info("TF-IDF vectorizer fitted successfully")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant documents for a query"""
        if not self.is_fitted:
            self.fit()
        
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant documents
                results.append({
                    'text': self.documents.iloc[idx]['text'],
                    'source_url': self.documents.iloc[idx].get('source_url', ''),
                    'similarity': similarities[idx],
                    'index': idx
                })
        
        logger.info(f"Retrieved {len(results)} relevant documents for query")
        return results
    
    def retrieve_by_keywords(self, keywords: List[str], k: int = 5) -> List[Dict]:
        """Retrieve documents containing specific keywords"""
        if self.documents is None:
            raise ValueError("No documents loaded")
        
        # Simple keyword matching
        mask = self.documents['text'].str.contains('|'.join(keywords), case=False, na=False)
        matching_docs = self.documents[mask].head(k)
        
        results = []
        for _, row in matching_docs.iterrows():
            results.append({
                'text': row['text'],
                'source_url': row.get('source_url', ''),
                'similarity': 1.0,  # Placeholder
                'keywords_found': keywords
            })
        
        logger.info(f"Retrieved {len(results)} documents matching keywords: {keywords}")
        return results
    
    def get_document_statistics(self) -> Dict:
        """Get statistics about the document collection"""
        if self.documents is None:
            return {}
        
        stats = {
            'total_documents': len(self.documents),
            'average_length': self.documents['text'].str.len().mean(),
            'total_words': self.documents['text'].str.split().str.len().sum(),
            'unique_sources': self.documents['source_url'].nunique() if 'source_url' in self.documents.columns else 0
        }
        
        return stats