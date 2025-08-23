"""
Neo4j Knowledge Graph creation and management
"""
from neo4j import GraphDatabase
import pandas as pd
import re
from typing import List, Dict, Tuple
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)

class KnowledgeGraph:
    """Neo4j Knowledge Graph manager for MAT data"""
    
    def __init__(self):
        self.driver = None
        self.connect()
        logger.info("KnowledgeGraph initialized")
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_entities_from_text(self, df: pd.DataFrame):
        """Extract and create entities from text data"""
        logger.info("Starting entity extraction and creation")
        
        entities = self._extract_entities(df)
        
        with self.driver.session() as session:
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    session.run(
                        f"MERGE (e:{entity_type} {{name: $name}})",
                        name=entity
                    )
        
        logger.info(f"Created entities: {sum(len(v) for v in entities.values())} total")
        return entities
    
    def _extract_entities(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract entities from text using simple pattern matching"""
        entities = {
            'MATStandard': [],
            'Organization': [],
            'Concept': [],
            'Location': []
        }
        
        # Patterns for different entity types
        mat_patterns = [
            r'MAT Standard \\d+',
            r'Standard \\d+',
            r'MAT \\d+'
        ]
        
        org_patterns = [
            r'Public Health Scotland',
            r'NHS',
            r'Government',
            r'Health Board'
        ]
        
        location_patterns = [
            r'Scotland',
            r'UK',
            r'United Kingdom'
        ]
        
        concept_patterns = [
            r'medication.assisted.treatment',
            r'substance.use',
            r'treatment',
            r'recovery',
            r'support'
        ]
        
        for _, row in df.iterrows():
            text = row['text'].lower()
            
            # Extract MAT Standards
            for pattern in mat_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['MATStandard'].extend(matches)
            
            # Extract Organizations
            for pattern in org_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    entities['Organization'].append(pattern)
            
            # Extract Locations
            for pattern in location_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    entities['Location'].append(pattern)
            
            # Extract Concepts
            for pattern in concept_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    entities['Concept'].append(pattern.replace('.', ' '))
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def create_relationships(self, df: pd.DataFrame):
        """Create relationships between entities"""
        logger.info("Creating relationships between entities")
        
        relationships = self._extract_relationships(df)
        
        with self.driver.session() as session:
            for rel_type, rel_list in relationships.items():
                for source, target in rel_list:
                    session.run(
                        f"""
                        MATCH (a {{name: $source}})
                        MATCH (b {{name: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        """,
                        source=source, target=target
                    )
        
        logger.info(f"Created {sum(len(v) for v in relationships.values())} relationships")
    
    def _extract_relationships(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
        """Extract relationships from text"""
        relationships = {
            'SUPPORTS': [],
            'IMPLEMENTS': [],
            'AIMS_AT': [],
            'PROVIDES': []
        }
        
        # Simple relationship extraction based on keywords
        for _, row in df.iterrows():
            text = row['text'].lower()
            
            if 'supports' in text:
                relationships['SUPPORTS'].append(('MAT', 'Support'))
            if 'implements' in text or 'implementation' in text:
                relationships['IMPLEMENTS'].append(('MAT Standards', 'Implementation'))
            if 'aims' in text or 'aimed at' in text:
                relationships['AIMS_AT'].append(('MAT Standards', 'Organizations'))
            if 'provides' in text:
                relationships['PROVIDES'].append(('MAT', 'Treatment'))
        
        return relationships
    
    def query_graph(self, query: str) -> List[Dict]:
        """Execute Cypher query on the graph"""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return []
    
    def get_related_entities(self, entity_name: str, relationship_type: str = None) -> List[Dict]:
        """Get entities related to a given entity"""
        if relationship_type:
            query = """
            MATCH (a {name: $entity_name})-[r:%s]->(b)
            RETURN b.name as related_entity, type(r) as relationship
            """ % relationship_type
        else:
            query = """
            MATCH (a {name: $entity_name})-[r]->(b)
            RETURN b.name as related_entity, type(r) as relationship
            """
        
        return self.query_graph(query)