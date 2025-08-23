"""
Setup script for the MAT Knowledge Graph Chatbot
"""
import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import config
from logger import setup_logger
from data_scraper import MATScraper
from knowledge_graph import KnowledgeGraph

logger = setup_logger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        config.RAW_DATA_PATH,
        config.PROCESSED_DATA_PATH,
        "logs",
        "models",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    logger.info("Installing Python requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_data():
    """Setup data by scraping and processing"""
    logger.info("Setting up data...")
    
    scraper = MATScraper()
    
    # Scrape data
    raw_data = scraper.scrape_all_urls()
    
    # Clean data
    cleaned_data = scraper.clean_data(raw_data)
    
    # Save data
    scraper.save_data(cleaned_data)
    
    logger.info("Data setup complete")

def setup_knowledge_graph():
    """Setup knowledge graph"""
    logger.info("Setting up knowledge graph...")
    
    import pandas as pd
    
    # Load cleaned data
    data_path = f"{config.PROCESSED_DATA_PATH}/{config.CLEANED_DATA_FILE}"
    df = pd.read_csv(data_path)
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    kg.create_entities_from_text(df)
    kg.create_relationships(df)
    kg.close()
    
    logger.info("Knowledge graph setup complete")

def train_rasa_model():
    """Train the Rasa model"""
    logger.info("Training Rasa model...")
    
    os.chdir(Path(__file__).parent.parent)
    subprocess.run(["rasa", "train"])
    
    logger.info("Rasa model training complete")

def main():
    """Main setup function"""
    logger.info("Starting MAT Chatbot setup...")
    
    try:
        create_directories()
        install_requirements()
        setup_data()
        setup_knowledge_graph()
        train_rasa_model()
        
        logger.info("Setup complete! You can now run the chatbot.")
        print("\\n" + "="*50)
        print("SETUP COMPLETE!")
        print("="*50)
        print("To start the chatbot:")
        print("1. Run: python -m src.chatbot")
        print("2. Or use Rasa: rasa run actions & rasa shell")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()