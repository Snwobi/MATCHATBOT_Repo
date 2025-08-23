"""
Logging configuration for the MAT Chatbot
"""
import logging
import os
from .config import config

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger