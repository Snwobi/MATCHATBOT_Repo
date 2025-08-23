"""
RAG (Retrieval-Augmented Generation) implementation with Llama2
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Optional
from .retriever import DocumentRetriever
from .knowledge_graph import KnowledgeGraph
from .config import config
from .logger import setup_logger

logger = setup_logger(__name__)

class RAGSystem:
    """RAG system combining document retrieval with language generation"""
    
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.kg = KnowledgeGraph()
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.is_loaded = False
        logger.info("RAGSystem initialized")
    
    def load_model(self):
        """Load the language model"""
        try:
            logger.info(f"Loading model: {config.MODEL_NAME}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=config.MAX_LENGTH,
                temperature=config.TEMPERATURE,
                do_sample=True,
                device=0 if device == "cuda" else -1
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Fallback to a smaller model or CPU
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller model as fallback"""
        try:
            logger.info("Loading fallback model")
            fallback_model = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                temperature=0.7
            )
            
            self.is_loaded = True
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
            self.is_loaded = False
    
    def setup(self):
        """Setup the RAG system"""
        logger.info("Setting up RAG system")
        
        # Load documents
        self.retriever.load_documents()
        self.retriever.fit()
        
        # Load model
        if not self.is_loaded:
            self.load_model()
        
        logger.info("RAG system setup complete")
    
    def generate_response(self, query: str, use_kg: bool = True, k: int = 3) -> Dict:
        """Generate response using RAG approach"""
        if not self.is_loaded:
            return {"response": "Model not loaded. Please try again later.", "sources": []}
        
        logger.info(f"Generating response for query: {query}")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query, k=k)
        
        # Get knowledge graph context if requested
        kg_context = ""
        if use_kg:
            kg_context = self._get_kg_context(query)
        
        # Prepare context for generation
        context = self._prepare_context(relevant_docs, kg_context)
        
        # Generate response
        response = self._generate_with_context(query, context)
        
        # Prepare final result
        result = {
            "response": response,
            "sources": [doc['source_url'] for doc in relevant_docs if doc.get('source_url')],
            "context_used": len(relevant_docs) > 0 or kg_context != "",
            "kg_context": kg_context != ""
        }
        
        logger.info("Response generated successfully")
        return result
    
    def _get_kg_context(self, query: str) -> str:
        """Get relevant context from knowledge graph"""
        try:
            # Simple keyword-based KG querying
            if "standard" in query.lower():
                kg_results = self.kg.query_graph(
                    "MATCH (n:MATStandard) RETURN n.name LIMIT 5"
                )
                if kg_results:
                    standards = [result['n.name'] for result in kg_results]
                    return f"Related MAT Standards: {', '.join(standards)}"
            
            if "organization" in query.lower():
                kg_results = self.kg.query_graph(
                    "MATCH (n:Organization) RETURN n.name LIMIT 3"
                )
                if kg_results:
                    orgs = [result['n.name'] for result in kg_results]
                    return f"Related Organizations: {', '.join(orgs)}"
            
        except Exception as e:
            logger.warning(f"KG context retrieval failed: {str(e)}")
        
        return ""
    
    def _prepare_context(self, relevant_docs: List[Dict], kg_context: str) -> str:
        """Prepare context for generation"""
        context_parts = []
        
        if kg_context:
            context_parts.append(f"Knowledge Graph Context: {kg_context}")
        
        if relevant_docs:
            doc_texts = [doc['text'][:200] + "..." for doc in relevant_docs[:3]]
            context_parts.append("Relevant Information: " + " ".join(doc_texts))
        
        return "\\n".join(context_parts)
    
    def _generate_with_context(self, query: str, context: str) -> str:
        """Generate response with given context"""
        try:
            # Prepare prompt
            prompt = f"""Context: {context}

Question: {query}

Answer: """
            
            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            response = generated_text.split("Answer: ")[-1].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove any trailing incomplete sentences
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Limit length
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response.strip()
    
    def get_system_info(self) -> Dict:
        """Get information about the RAG system"""
        return {
            "model_loaded": self.is_loaded,
            "model_name": config.MODEL_NAME if self.is_loaded else "None",
            "retriever_fitted": self.retriever.is_fitted,
            "document_stats": self.retriever.get_document_statistics(),
            "kg_connected": self.kg.driver is not None
        }