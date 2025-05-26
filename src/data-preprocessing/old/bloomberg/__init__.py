"""
Bloomberg Data Processing Module

This package contains modules for downloading and processing Bloomberg news data.
Features include:
- Data downloading from Bloomberg
- Feature extraction from headlines
- Sentiment analysis
- Event classification
"""

from .bloomberg_feature_extractor import (
    process_dataset,
    create_feature_columns,
    format_prompt,
    parse_model_response
)
from .bloomberg_data_download import *

# Version of the Bloomberg module
__version__ = "0.1.0"

# Public API
__all__ = [
    'process_dataset',
    'create_feature_columns',
    'format_prompt',
    'parse_model_response'
] 