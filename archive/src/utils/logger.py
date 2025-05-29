import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_name: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    Creates a new log file for each run in the logs directory.
    
    Args:
        log_name (str, optional): Name of the logger. If None, uses the module name.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create a new log file for each run with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'pipeline_{timestamp}.log'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Get logger for the specific module
    logger = logging.getLogger(log_name)
    
    # Log the initialization
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger 