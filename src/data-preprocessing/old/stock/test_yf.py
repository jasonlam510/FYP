"""
Yahoo Finance Data Downloader

This script downloads financial data from Yahoo Finance for specified symbols and time periods.
It supports various intervals (daily, hourly, etc.) and saves the data in CSV format.

The script:
1. Downloads financial data from Yahoo Finance
2. Saves the raw data to a specified location
3. Provides command-line argument support for customization

Usage:
    Basic usage with default parameters:
        python test_yf.py

    Download specific stock with custom date range:
        python test_yf.py --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01

    Download hourly data:
        python test_yf.py --symbol ^SPX --interval 1h

    Specify custom output path:
        python test_yf.py --output-path /path/to/your/data.csv

Configuration:
    Default parameters:
        - symbol: ^SPX (S&P 500 index)
        - start_date: 2010-01-01
        - end_date: 2024-12-01
        - interval: 1d (daily data)
        - output_path: data/raw/yf/{symbol}_{interval}.csv

    Available intervals:
        - 1d: Daily data
        - 1h: Hourly data
        - 1wk: Weekly data
        - 1mo: Monthly data
        - 1m: 1-minute data (limited to 7 days)
        - 5m: 5-minute data (limited to 60 days)
        - 15m: 15-minute data (limited to 60 days)
        - 30m: 30-minute data (limited to 60 days)
        - 90m: 90-minute data (limited to 60 days)

Note:
    - For intraday data (1m, 5m, etc.), the date range is limited
    - Some symbols might require '^' prefix for indices (e.g., ^SPX, ^DJI)
    - The script will automatically create necessary directories
    - If data already exists at the output path, it will be loaded instead of downloaded
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
import argparse

def create_directory(directory_path: Path) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path (Path): Path to the directory to create
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory created/verified: {directory_path}")

def download_finance_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    output_path: str = None
) -> pd.DataFrame:
    """Download financial data from Yahoo Finance and save it locally.
    
    Args:
        symbol (str): Stock symbol (e.g., '^SPX')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d' for daily, '1h' for hourly, etc.)
        output_path (str, optional): Custom path to save the data. If None, uses default path.
    
    Returns:
        pd.DataFrame: The downloaded data
    """
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Set default output path if not provided
    if output_path is None:
        output_path = script_dir.parent.parent.parent / "data" / "raw" / "yf" / f"{symbol.replace('^', '')}_{interval}_new.csv"
    else:
        output_path = Path(output_path)
    
    # Create directory for the output path
    create_directory(output_path.parent)
    
    # Check if the data is already downloaded
    if output_path.exists():
        print(f"Data already exists at {output_path}")
        print("Loading existing data...")
        df = pd.read_csv(output_path)
    else:
        # Download data from Yahoo Finance
        print(f"Downloading {interval} data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Save the data
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    # Print data information
    print(f"\nData shape: {df.shape}")
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    return df

def main():
    """Main function to execute the data downloading process."""
    parser = argparse.ArgumentParser(description='Download financial data from Yahoo Finance')
    parser.add_argument('--symbol', type=str, default='^SPX', help='Stock symbol (e.g., ^SPX)')
    parser.add_argument('--start-date', type=str, default='2006-10-20', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2013-11-26', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1d for daily, 1h for hourly)')
    parser.add_argument('--output-path', type=str, help='Custom path to save the data')
    args = parser.parse_args()
    
    try:
        df = download_finance_data(
            args.symbol,
            args.start_date,
            args.end_date,
            args.interval,
            args.output_path
        )
        print("\nFirst few rows of the data:")
        print(df.head())
        print("\nData downloading completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
