import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import sys
import os
import logging
from sklearn.preprocessing import OneHotEncoder

current_dir = Path.cwd()
project_root = current_dir
sys.path.append(str(project_root))

from src.config.data_processor_config import DATA_FREQUENCY, NEWS_AGGREGATION, EVENT_TYPES
from src.utils.data_format import convert_datetime_column, standardize_dates
from src.config.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Output configuration
OUTPUT_DIR = 'data/processed/pipeline'
DEBUG_DIR = 'data/debug' 

def debug_df(df: pd.DataFrame, name: str):
    # Save the df to debug
    debug_file_path = os.path.join(DEBUG_DIR, f'{name}.csv')
    df.to_csv(debug_file_path, index=False)
    logger.info(f"Debug {name} DataFrame saved to {debug_file_path}")

class DataProcessor:
    def __init__(self, window_size: int = 10):
        """
        Initialize the data processor.
        
        Args:
            window_size (int): Number of days to include in each sample window
        """
        self.window_size = window_size
        self.onehot = OneHotEncoder(categories=[EVENT_TYPES], sparse_output=False, handle_unknown='ignore')
        logger.info(f"DataProcessor initialized with window_size={window_size}")

    def _find_date_overlap(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find the overlapping date range between price and news data.
        Aligns news data with trading days - news on non-trading days are counted towards next trading day.
        
        Args:
            price_df (pd.DataFrame): Price data (trading days)
            news_df (pd.DataFrame): News data
            
        Returns:
            Tuple of price and news DataFrames with overlapping dates
        """
        # Convert dates to datetime using data_format utility
        price_df = convert_datetime_column(price_df, 'date')
        news_df = convert_datetime_column(news_df, 'date')
        
        # Get all trading days
        trading_dates = sorted(price_df['date'].dt.date.unique())
        
        # Create a mapping of dates to next trading day
        date_to_next_trading = {}
        for i in range(len(trading_dates) - 1):
            current_trading = trading_dates[i]
            next_trading = trading_dates[i + 1]
            date_to_next_trading[current_trading] = current_trading
            # Fill in all dates between current and next trading day
            current = current_trading
            while current < next_trading:
                current = (pd.Timestamp(current) + pd.Timedelta(days=1)).date()
                date_to_next_trading[current] = next_trading

        # Map news dates to next trading day
        news_df['trading_date'] = news_df['date'].dt.date.map(lambda x: date_to_next_trading.get(x, x))

        debug_df(news_df, 'news-df-with-trading-dates')
        
        # Filter news data to only include dates that map to trading days
        news_df = news_df[news_df['trading_date'].isin(trading_dates)]
        
        # Get date ranges for logging
        min_date = min(trading_dates)
        max_date = max(trading_dates)
        logger.info(f"Trading date range: {min_date} to {max_date}")
        logger.info(f"Found {len(trading_dates)} trading days")
        
        return price_df, news_df

    def _aggregate_news_daily(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news features to daily frequency and add to stock data.
        For daily data: aggregates all news for each trading day
        For hourly data: TODO - will implement hourly aggregation
        
        Args:
            news_df (pd.DataFrame): News data with multiple entries per day
            
        Returns:
            pd.DataFrame: Daily aggregated news data with sentiment and event features
        """
        logger.info(f"Aggregating news data to {DATA_FREQUENCY} frequency...")
        
        # One-hot encode event_type column
        if 'event_type' in news_df.columns:
            # Convert event_type to string type
            news_df['event_type'] = news_df['event_type'].astype(str)
            # Fit and transform the event_type column
            event_encoded = self.onehot.fit_transform(news_df[['event_type']])
            # Create DataFrame with encoded event types
            event_df = pd.DataFrame(event_encoded, 
                                  columns=[f'event_type_{event}' for event in EVENT_TYPES],
                                  index=news_df.index)
            # Concatenate with original DataFrame
            news_df = pd.concat([news_df, event_df], axis=1)
            logger.info("One-hot encoded event_type column")
        
        if DATA_FREQUENCY == 'daily':
            # Group by date and aggregate in one operation
            agg_dict = {}
            for col, method in NEWS_AGGREGATION.items():
                if col in news_df.columns:
                    if method == 'mean':
                        agg_dict[col] = 'mean'  # Average sentiment, relevance, importance
                    elif method == 'count':
                        if col == 'event_type':
                            # For event type columns, sum the one-hot encoded values
                            event_cols = [col for col in news_df.columns if col.startswith('event_type_')]
                            for event_col in event_cols:
                                agg_dict[event_col] = 'sum'  # Count of events by type
                        else:
                            agg_dict[col] = 'count'  # Count of other features
            
            # Perform aggregation
            agg_df = news_df.groupby('trading_date').agg(agg_dict)
            agg_df.reset_index(inplace=True)
            
            # Rename trading_date back to date for merging with price data
            agg_df.rename(columns={'trading_date': 'date'}, inplace=True)
            
            debug_df(agg_df, 'aggregated-news-df')
            
            # Fill missing event type columns with 0
            event_cols = [col for col in agg_df.columns if col.startswith('event_type_')]
            if event_cols:
                agg_df[event_cols] = agg_df[event_cols].fillna(0)
            
            logger.info(f"News aggregation completed. Shape: {agg_df.shape}")
            return agg_df
            
        elif DATA_FREQUENCY == 'hourly':
            # TODO: Implement hourly aggregation
            raise NotImplementedError("Hourly aggregation not implemented yet")

    def combine_data(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine pre-processed price and news data.
        
        Args:
            price_df (pd.DataFrame): Pre-processed price data with technical indicators
            news_df (pd.DataFrame): Pre-processed news data with sentiment and event features
            
        Returns:
            pd.DataFrame: Combined dataset ready for LSTM training
        """
        logger.info("Starting data combination...")
        
        # Find overlapping dates
        price_df, news_df = self._find_date_overlap(price_df, news_df)
        
        # Aggregate news data to daily frequency
        news_agg = self._aggregate_news_daily(news_df)
        
        # Ensure date columns are in the same format using data_format utilities
        price_df = convert_datetime_column(price_df, 'date')
        news_agg = convert_datetime_column(news_agg, 'date')
        
        # Convert to date objects for merging
        price_df['date'] = price_df['date'].dt.date
        news_agg['date'] = news_agg['date'].dt.date
        
        # Merge price and news data on date
        combined_df = pd.merge(price_df, news_agg, on='date', how='left')
        
        # Fill missing news features with 0 (no news on that day)
        news_columns = [col for col in combined_df.columns if col not in price_df.columns]
        combined_df[news_columns] = combined_df[news_columns].fillna(0)
        
        logger.info(f"Data combination completed. Final shape: {combined_df.shape}")
        return combined_df

def run_data_processor(price_df: pd.DataFrame, news_df: pd.DataFrame, window_size: int = 20, output_csv_path: str = None) -> pd.DataFrame:
    """
    End-to-end data processing pipeline.
    
    Args:
        price_df (pd.DataFrame): Pre-processed price data
        news_df (pd.DataFrame): Pre-processed news data
        window_size (int): Number of days to include in each sample window
        output_csv_path (str, optional): Path to save the combined data as CSV
        
    Returns:
        pd.DataFrame: Combined dataset ready for LSTM training
    """
    logger.info("Starting data processing pipeline...")
    
    processor = DataProcessor(window_size=window_size)
    combined_df = processor.combine_data(price_df, news_df)
    
    # Save combined data to CSV if path is provided
    if output_csv_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Export to CSV
        combined_df.to_csv(output_csv_path, index=False)
        logger.info(f"Combined data saved to {output_csv_path}")
        logger.info(f"Number of features: {len(combined_df.columns)}")
        logger.info(f"Number of samples: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    logger.info("Data processing pipeline completed successfully")
    return combined_df

def test():
    """
    Test function to demonstrate pipeline usage.
    """
    # Load pre-processed data
    logger.info("Loading pre-processed data...")
    price_df = pd.read_csv('data/pipeline/processed_price_data.csv')
    news_df = pd.read_csv('data/pipeline/feature_extracted_data.csv')
    
    # Run data processor
    combined_df = run_data_processor(price_df, news_df, output_csv_path='data/debug/combined_data.csv')
    
    logger.info("\nData shapes:")
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Save processed data
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    feature_cols = [col for col in combined_df.columns if col != 'date']
    X = combined_df[feature_cols].values
    y = combined_df['close'].values
    dates = combined_df['date'].values
    
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    np.save(output_dir / 'dates.npy', dates)
    logger.info(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    test() 