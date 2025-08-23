#!/usr/bin/env python3
"""
MAT Standards Chatbot - Main Entry Point
AI-Driven Medication-Assisted Treatment Chatbot for Scottish Healthcare
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mat_chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USER', 
        'NEO4J_PASSWORD',
        'HUGGINGFACE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        return False
    
    return True

def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        MAT Standards Chatbot for Scottish Healthcare         ║
    ║                                                              ║
    ║        🏥 AI-Driven Healthcare Assistant                     ║
    ║        🤖 Powered by Llama2 + Neo4j + RAG                   ║
    ║        📋 Medication-Assisted Treatment Support              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_streamlit_app():
    """Launch the Streamlit web interface."""
    try:
        import streamlit.web.cli as stcli
        import streamlit as st
        
        # Check if streamlit app exists
        app_path = project_root / "src" / "interface" / "streamlit_app.py"
        if app_path.exists():
            print(f"🚀 Launching Streamlit interface...")
            print(f"📂 App location: {app_path}")
            
            # Run streamlit app
            sys.argv = ["streamlit", "run", str(app_path), "--server.port=8501"]
            stcli.main()
        else:
            print(f"❌ Streamlit app not found at: {app_path}")
            print("Please ensure src/interface/streamlit_app.py exists")
            return False
            
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    
    return True

def run_rasa_server():
    """Launch the RASA server for conversational AI."""
    try:
        import subprocess
        
        print("🤖 Starting RASA server...")
        
        # Start RASA actions server
        actions_process = subprocess.Popen([
            "rasa", "run", "actions", "--port", "5055"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Start main RASA server
        rasa_process = subprocess.Popen([
            "rasa", "run", "--port", "5005", "--enable-api", "--cors", "*"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ RASA server started on port 5005")
        print("✅ RASA actions server started on port 5055")
        
        return True
        
    except Exception as e:
        print(f"❌ Error starting RASA server: {e}")
        return False

def test_connections():
    """Test connections to external services."""
    print("🔍 Testing system connections...")
    
    # Test Neo4j connection
    try:
        from src.chatbot.knowledge_graph import Neo4jConnection
        
        neo4j_conn = Neo4jConnection(
            uri=os.getenv('NEO4J_URI'),
            user=os.getenv('NEO4J_USER'),
            pwd=os.getenv('NEO4J_PASSWORD')
        )
        
        # Simple test query
        result = neo4j_conn.query("RETURN 1 as test")
        if result:
            print("✅ Neo4j connection successful")
        else:
            print("❌ Neo4j connection failed")
            return False
            
        neo4j_conn.close()
        
    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        return False
    
    # Test LLM availability
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        print("✅ Llama2 model accessible")
        
    except Exception as e:
        print(f"❌ Llama2 model error: {e}")
        return False
    
    return True

def main():
    """Main application entry point."""
    # Setup
    logger = setup_logging()
    print_banner()
    
    logger.info("Starting MAT Chatbot application...")
    
    # Environment check
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # Test connections
    if not test_connections():
        logger.error("Connection tests failed")
        sys.exit(1)
    
    # Show menu
    print("\n🎯 Choose an option:")
    print("1. Launch Streamlit Web Interface")
    print("2. Start RASA Conversational Server")
    print("3. Run System Tests")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🌐 Starting web interface...")
                if run_streamlit_app():
                    break
            
            elif choice == "2":
                print("\n💬 Starting conversational AI server...")
                if run_rasa_server():
                    print("Press Ctrl+C to stop the server")
                    try:
                        while True:
                            pass
                    except KeyboardInterrupt:
                        print("\n🛑 Server stopped")
                        break
            
            elif choice == "3":
                print("\n🧪 Running comprehensive system tests...")
                # Add your test functions here
                from tests.test_knowledge_graph import test_neo4j_queries
                from tests.test_llm import test_llama_response
                from tests.test_rag import test_rag_pipeline
                
                print("Running Neo4j tests...")
                test_neo4j_queries()
                
                print("Running LLM tests...")
                test_llama_response()
                
                print("Running RAG pipeline tests...")
                test_rag_pipeline()
                
                print("✅ All tests completed!")
            
            elif choice == "4":
                print("\n👋 Goodbye! Thank you for using MAT Chatbot!")
                logger.info("Application terminated by user")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Application interrupted by user")
            logger.info("Application interrupted")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}")
        sys.exit(1)