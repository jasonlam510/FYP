import logging
import os
from datetime import datetime

def get_logger(name):
    """
    Get a logger instance with both file and console handlers.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Get job name from environment variable or use default
    job_name = os.getenv('JOB_NAME', 'default_job')
    
    # Create a logger
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added before
    if not logger.handlers:
        # Set the logging level
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create file handler
        log_filename = f'logs/{job_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log the initialization
        logger.info(f"Logger initialized for {name}")
        logger.info(f"Log file: {log_filename}")
    
    return logger
