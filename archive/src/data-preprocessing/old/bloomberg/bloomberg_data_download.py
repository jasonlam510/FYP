"""
Bloomberg Financial News Dataset Downloader

This script downloads and processes the Bloomberg Financial News Dataset (2006-2013) 
for our event-based stock prediction project. The dataset contains 446,762 financial news 
articles that will be used to train our prediction model.

The script:
1. Downloads the Bloomberg Financial News dataset from Hugging Face
2. Saves the raw data to a specified directory (default: /data/raw/bloomberg)
3. Provides command-line argument support for custom output paths
4. Performs basic data validation and information display

Usage:
    python bloomberg_data_download.py
    python bloomberg_data_download.py --output-path /custom/path

Data Source: https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News
"""

import pandas as pd
import os
from pathlib import Path
import argparse

# Constants
# BLOOMBERG_DATASET_URL = "hf://datasets/danidanou/Bloomberg_Financial_News/bloomberg_financial_data.parquet.gzip"
BLOOMBERG_DATASET_URL = "hf://datasets/danidanou/Reuters_Financial_News/summ_financial_data.parquet.gzip"
DEFAULT_OUTPUT_FILENAME = "reuters_financial_data.parquet"
DEFAULT_OUTPUT_SUBPATH = Path("data") / "raw" / "bloomberg"

def create_directory(directory_path: Path) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path (Path): Path to the directory to create
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory created/verified: {directory_path}")

def load_and_save_dataset(output_path: Path = None) -> pd.DataFrame:
    """Load the Bloomberg dataset from Hugging Face and save it locally.
    
    Args:
        output_path (Path, optional): Custom path to save the dataset. If None, uses default path.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Set default output path if not provided
    if output_path is None:
        output_path = script_dir.parent.parent.parent / DEFAULT_OUTPUT_SUBPATH / DEFAULT_OUTPUT_FILENAME
    else:
        output_path = Path(output_path) / DEFAULT_OUTPUT_FILENAME
    
    # Create directory for the output path
    create_directory(output_path.parent)
    
    # Check if the dataset is already downloaded
    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
        print("Loading existing dataset...")
        df = pd.read_parquet(output_path, engine='auto', date_format='ISO8601')
    else:
        # Load the dataset from Hugging Face
        print("Loading dataset from Hugging Face...")
        df = pd.read_parquet(BLOOMBERG_DATASET_URL, engine='auto', date_format='ISO8601')
        
        # Save the raw data
        df.to_parquet(output_path)
        print(f"Dataset saved to {output_path}")
    
    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    return df

def main():
    """Main function to execute the data loading and saving process."""
    parser = argparse.ArgumentParser(description='Download and process Bloomberg Financial News dataset')
    parser.add_argument('--output-path', type=str, help='Custom path to save the dataset')
    args = parser.parse_args()
    
    try:
        df = load_and_save_dataset(args.output_path)
        print("Data loading and saving completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
