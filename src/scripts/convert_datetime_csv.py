"""
Script to convert datetime columns in a CSV file to ISO 8601 format with specified timezone.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path.cwd()
sys.path.append(str(project_root))

from src.utils.data_format import convert_datetime_column
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

def convert_csv_datetime(input_path: str, output_path: str, datetime_columns: list, target_tz: str = 'US/Eastern'):
    """
    Convert datetime columns in a CSV file to ISO 8601 format with specified timezone.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save converted CSV file
        datetime_columns (list): List of column names containing datetime data
        target_tz (str): Target timezone. Defaults to 'US/Eastern'
    """
    try:
        # Read the CSV file
        logger.info(f"Reading CSV file from {input_path}")
        df = pd.read_csv(input_path)
        
        # Convert each datetime column
        for col in datetime_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in CSV file")
                continue
                
            logger.info(f"Converting datetime column: {col}")
            df = convert_datetime_column(df, col, target_tz)
        
        # Save the converted data
        logger.info(f"Saving converted data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert datetime columns in a CSV file to ISO 8601 format')
    parser.add_argument('input_path', help='Path to input CSV file')
    parser.add_argument('output_path', help='Path to save converted CSV file')
    parser.add_argument('--columns', nargs='+', required=True, help='List of datetime column names to convert')
    parser.add_argument('--timezone', default='US/Eastern', help='Target timezone (default: US/Eastern)')
    
    args = parser.parse_args()
    
    convert_csv_datetime(
        input_path=args.input_path,
        output_path=args.output_path,
        datetime_columns=args.columns,
        target_tz=args.timezone
    )

def test():
    input_path = "data/processed/yf/SPX_1d_with_indicators.csv"
    output_path = "data/processed/yf/SPX_1d_with_indicators_converted.csv"
    datetime_columns = ["date"]
    target_tz = "US/Eastern"
    convert_csv_datetime(input_path, output_path, datetime_columns, target_tz)

if __name__ == '__main__':
    test() 

 