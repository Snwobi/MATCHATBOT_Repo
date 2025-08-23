"""
Main chatbot application with Gradio interface
"""
import gradio as gr
import pandas as pd
from typing import Dict, Tuple
from .rag_system import RAGSystem
from .data_scraper import MATScraper
from .knowledge_graph import KnowledgeGraph
from .config import config
from .logger import setup_logger
import os

logger = setup_logger(__name__)

class MATChatbot:
    """Main MAT Chatbot application"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.scraper = MATScraper()
        self.kg = KnowledgeGraph()
        self.chat_history = []
        logger.info("MATChatbot initialized")
    
    def setup_system(self):
        """Setup the entire chatbot system"""
        logger.info("Setting up MAT Chatbot system")
        
        # Check if data exists, if not scrape it
        data_path = f"{config.PROCESSED_DATA_PATH}/{config.CLEANED_DATA_FILE}"
        if not os.path.exists(data_path):
            logger.info("Data not found, starting scraping process")
            self.scrape_and_process_data()
        
        # Setup RAG system
        self.rag_system.setup()
        
        logger.info("System setup complete")
    
    def scrape_and_process_data(self):
        """Scrape and process data from MAT websites"""
        logger.info("Starting data scraping and processing")
        
        # Scrape data
        raw_data = self.scraper.scrape_all_urls()
        
        # Clean data
        cleaned_data = self.scraper.clean_data(raw_data)
        
        # Save data
        self.scraper.save_data(cleaned_data)
        
        # Create knowledge graph
        self.kg.create_entities_from_text(cleaned_data)
        self.kg.create_relationships(cleaned_data)
        
        logger.info("Data scraping and processing complete")
    
    def process_query(self, query: str) -> Tuple[str, str]:
        """Process user query and return response with chat history"""
        if not query.strip():
            return "", self._format_chat_history()
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Generate response using RAG
            result = self.rag_system.generate_response(query)
            response = result['response']
            
            # Add to chat history
            self.chat_history.append({
                'query': query,
                'response': response,
                'sources': result.get('sources', [])
            })
            
            # Keep only last 10 exchanges
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return "", self._format_chat_history()
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            error_response = "I apologize, but I encountered an error. Please try again."
            
            self.chat_history.append({
                'query': query,
                'response': error_response,
                'sources': []
            })
            
            return "", self._format_chat_history()
    
    def _format_chat_history(self) -> str:
        """Format chat history for display"""
        if not self.chat_history:
            return "Welcome! Ask me anything about MAT (Medication-Assisted Treatment) standards."
        
        formatted_history = []
        for exchange in self.chat_history:
            formatted_history.append(f"**You:** {exchange['query']}")
            formatted_history.append(f"**MAT Bot:** {exchange['response']}")
            
            if exchange.get('sources'):
                sources_text = "📚 Sources: " + ", ".join(exchange['sources'][:2])
                formatted_history.append(f"*{sources_text}*")
            
            formatted_history.append("")  # Empty line for spacing
        
        return "\\n".join(formatted_history)
    
    def clear_chat(self) -> Tuple[str, str]:
        """Clear chat history"""
        self.chat_history = []
        return "", "Chat cleared! How can I help you with MAT standards?"
    
    def get_system_status(self) -> str:
        """Get system status information"""
        try:
            info = self.rag_system.get_system_info()
            
            status = f"""
## System Status
- **Model Status:** {'✅ Loaded' if info['model_loaded'] else '❌ Not Loaded'}
- **Documents Loaded:** {info['document_stats'].get('total_documents', 0)}
- **Knowledge Graph:** {'✅ Connected' if info['kg_connected'] else '❌ Disconnected'}
- **Average Document Length:** {info['document_stats'].get('average_length', 0):.0f} characters
- **Total Words:** {info['document_stats'].get('total_words', 0):,}
"""
            return status
            
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return "❌ Unable to retrieve system status"
    
    def launch_interface(self, share: bool = True):
        """Launch the Gradio interface"""
        logger.info("Launching Gradio interface")
        
        # Setup system before launching
        self.setup_system()
        
        # Create Gradio interface
        with gr.Blocks(title="MAT Knowledge Chatbot", theme=gr.themes.Soft()) as iface:
            gr.Markdown(
                """
                # 🏥 MAT Knowledge Chatbot
                ### Medication-Assisted Treatment Standards Information System
                
                Ask me anything about MAT standards, implementation, or related topics!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot_display = gr.Textbox(
                        label="Chat History",
                        value="Welcome! Ask me anything about MAT (Medication-Assisted Treatment) standards.",
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What is MAT Standard 1?",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        status_btn = gr.Button("System Status", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ### 💡 Example Questions
                        - What are the MAT standards?
                        - Tell me about MAT Standard 4
                        - How are MAT standards implemented?
                        - What organizations are involved in MAT?
                        - What is the aim of MAT?
                        
                        ### 🔧 Features
                        - **Knowledge Graph Integration**
                        - **Document Retrieval**
                        - **AI-Powered Responses**
                        - **Source Citations**
                        """
                    )
                    
                    system_status = gr.Textbox(
                        label="System Information",
                        lines=8,
                        interactive=False
                    )
            
            # Event handlers
            def handle_query(query, history):
                return self.process_query(query)
            
            def handle_clear():
                return self.clear_chat()
            
            def handle_status():
                return self.get_system_status()
            
            # Connect events
            send_btn.click(
                fn=handle_query,
                inputs=[query_input, chatbot_display],
                outputs=[query_input, chatbot_display]
            )
            
            query_input.submit(
                fn=handle_query,
                inputs=[query_input, chatbot_display],
                outputs=[query_input, chatbot_display]
            )
            
            clear_btn.click(
                fn=handle_clear,
                outputs=[query_input, chatbot_display]
            )
            
            status_btn.click(
                fn=handle_status,
                outputs=system_status
            )
            
            # Load initial status
            iface.load(
                fn=handle_status,
                outputs=system_status
            )
        
        # Launch interface
        iface.launch(
            share=share,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
        
        logger.info("Gradio interface launched")

if __name__ == "__main__":
    chatbot = MATChatbot()
    chatbot.launch_interface()