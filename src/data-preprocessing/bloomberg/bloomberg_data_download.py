"""
Bloomberg Financial News Dataset Preprocessing

This script handles the preprocessing of the Bloomberg Financial News Dataset (2006-2013) 
for our event-based stock prediction project. The dataset contains 446,762 financial news 
articles that will be used to train our prediction model.

The script:
1. Loads the Bloomberg Financial News dataset from Hugging Face
2. Saves the raw data to our local directory structure
3. Performs initial data exploration and preprocessing steps

Data Source: https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News
"""

import pandas as pd
import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def create_directory(directory_path: Path) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path (Path): Path to the directory to create
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created/verified: {directory_path}")

def load_and_save_dataset() -> pd.DataFrame:
    """Load the Bloomberg dataset from Hugging Face and save it locally.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Create directory for raw data (relative to script location)
    raw_data_dir = script_dir.parent.parent.parent / "data" / "raw" / "bloomberg"
    create_directory(raw_data_dir)
    
    # Define the output path
    output_path = raw_data_dir / "bloomberg_financial_data.parquet"
    
    # Check if the dataset is already downloaded
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        logger.info("Loading existing dataset...")
        df = pd.read_parquet(output_path)
    else:
        # Load the dataset from Hugging Face
        logger.info("Loading dataset from Hugging Face...")
        df = pd.read_parquet("hf://datasets/danidanou/Bloomberg_Financial_News/bloomberg_financial_data.parquet.gzip")
        
        # Save the raw data
        df.to_parquet(output_path)
        logger.info(f"Dataset saved to {output_path}")
    
    # Log dataset information
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("Columns in the dataset:")
    logger.info(df.columns.tolist())
    
    return df

def main():
    """Main function to execute the data loading and saving process."""
    try:
        df = load_and_save_dataset()
        logger.info("Data loading and saving completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
