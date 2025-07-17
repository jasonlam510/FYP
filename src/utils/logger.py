import logging
import os
from datetime import datetime

class LoggerManager:
    _instance = None
    _loggers = {}
    _current_job = None
    _current_log_file = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_logger(self, name):
        """
        Get a logger instance with both file and console handlers.
        Uses a single log file per job run, shared across all modules.
        
        Args:
            name: Name of the logger (typically __name__ of the calling module)
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Get job name from environment variable or use default
        job_name = os.getenv('JOB_NAME', 'default_job')
        
        # If this is a new job, reset the log file
        if job_name != self._current_job:
            self._current_job = job_name
            self._current_log_file = None
            self._loggers.clear()
        
        # Create a logger
        logger = logging.getLogger(name)
        
        # If this logger already exists, return it
        if name in self._loggers:
            return logger
        
        # Set the logging level
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create file handler - only if this is the first logger for this job
        if self._current_log_file is None:
            timestamp = os.environ.get('START_TIME', datetime.now().strftime("%Y%m%d_%H%M%S"))
            self._current_log_file = f'logs/{job_name}_{timestamp}.log'
            file_handler = logging.FileHandler(self._current_log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Created new log file: {self._current_log_file}")
        else:
            # Reuse existing file handler
            file_handler = logging.FileHandler(self._current_log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Create console handler if it doesn't exist
        if not any(h for h in logger.handlers if isinstance(h, logging.StreamHandler)):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Store this logger
        self._loggers[name] = logger
        
        return logger

# Create a singleton instance
_logger_manager = LoggerManager()

def get_logger(name):
    """
    Get a logger instance using the singleton manager.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return _logger_manager.get_logger(name)
