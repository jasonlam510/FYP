import logging
import os
from pathlib import Path

def setup_logging(
    level=logging.INFO,
    log_format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    log_file: str = None
):
    """
    Set up logging configuration for the project.

    Args:
        level (int): Logging level (default: logging.INFO)
        log_format (str): Format for log messages
        log_file (str, optional): If provided, log messages will also be written to this file
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        # Ensure the log folder is in the project root
        project_root = Path(__file__).resolve().parents[2]
        log_path = project_root / log_file
        os.makedirs(log_path.parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # Overwrite any existing logging config
    ) 