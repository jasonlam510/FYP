import pandas as pd
import numpy as np
import sys
from pathlib import Path
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import CCIIndicator
from typing import List, Dict, Union
import logging

current_dir = Path.cwd()
project_root = current_dir
sys.path.append(str(project_root))
from src.config.stock_pipeline_config import INDICATORS
from src.utils.data_format import standardize_date, convert_datetime_column
from src.config.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

class PricePipeline:
    def __init__(self):
        """
        Initialize the price pipeline with technical indicators.
        """
        self.indicators = INDICATORS
        self.required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        logger.info("PricePipeline initialized")

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame schema and data types.
        
        Args:
            df (pd.DataFrame): Input price data
            
        Raises:
            ValueError: If required columns are missing or data types are invalid
        """
        # Lowercase the column names
        df.columns = df.columns.str.lower()
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate data types
        try:
            df = convert_datetime_column(df, 'date')
            logger.info("Successfully converted date column to datetime format")
        except Exception as e:
            logger.error(f"Failed to convert date column: {str(e)}")
            raise ValueError("Column 'date' must be convertible to datetime")
                
        # Validate numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column '{col}' must be numeric")
                raise ValueError(f"Column '{col}' must be numeric")
            if (df[col] < 0).any():
                negative_rows = df[df[col] < 0]
                logger.warning(f"Column '{col}' contains negative values in {len(negative_rows)} rows")
                logger.debug(f"Negative values in {col}:\n{negative_rows}")
                raise ValueError(f"Column '{col}' contains negative values")
                
        # Validate price relationships
        incorrect_rows = df[~(df['high'] >= df['low'])]
        if not incorrect_rows.empty:
            logger.error(f"Found {len(incorrect_rows)} rows where high price is less than low price")
            logger.debug(f"Incorrect row indices: {incorrect_rows.index.tolist()}")
            raise ValueError("High price cannot be less than low price")
        
        incorrect_rows = df[~((df['high'] >= df['open']) & (df['high'] >= df['close']))]
        if not incorrect_rows.empty:
            logger.error(f"Found {len(incorrect_rows)} rows where high price is less than open or close")
            logger.debug(f"Incorrect row indices: {incorrect_rows.index.tolist()}")
            raise ValueError("High price must be greater than or equal to open and close")
        
        incorrect_rows = df[~((df['low'] <= df['open']) | (df['low'] <= df['close']))]
        if not incorrect_rows.empty:
            logger.error(f"Found {len(incorrect_rows)} rows where low price is greater than open or close")
            logger.debug(f"Incorrect row indices: {incorrect_rows.index.tolist()}")
            raise ValueError("Low price must be less than or equal to open and close")
            
        logger.info("Input validation completed successfully")

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given price data using ta library.
        
        Args:
            df (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with calculated indicators
        """
        logger.info("Calculating technical indicators...")
        
        for indicator in self.indicators:
            name = indicator['name']
            window = indicator.get('window', None)
            logger.debug(f"Calculating {name} indicator with window {window}")
            
            try:
                if name == 'bb':
                    bb = BollingerBands(close=df['close'], window=window)
                    df[f'bb_upper_{window}'] = bb.bollinger_hband()
                    df[f'bb_middle_{window}'] = bb.bollinger_mavg()
                    df[f'bb_lower_{window}'] = bb.bollinger_lband()
                    
                elif name == 'ma':
                    ma = SMAIndicator(close=df['close'], window=window)
                    df[f'ma_{window}'] = ma.sma_indicator()
                    
                elif name == 'ema':
                    ema = EMAIndicator(close=df['close'], window=window)
                    df[f'ema_{window}'] = ema.ema_indicator()
                    
                elif name == 'rsi':
                    rsi = RSIIndicator(close=df['close'], window=window)
                    df[f'rsi_{window}'] = rsi.rsi()
                    
                elif name == 'macd':
                    macd = MACD(close=df['close'], window_slow=window, window_fast=12, window_sign=9)
                    df[f'macd_{window}'] = macd.macd()
                    df[f'macd_signal_{window}'] = macd.macd_signal()
                    df[f'macd_hist_{window}'] = macd.macd_diff()
                    
                elif name == 'atr':
                    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
                    df[f'atr_{window}'] = atr.average_true_range()
                    
                elif name == 'cci':
                    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
                    df[f'cci_{window}'] = cci.cci()
                    
                elif name == 'stochastic':
                    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=window)
                    df[f'stoch_k_{window}'] = stoch.stoch()
                    df[f'stoch_d_{window}'] = stoch.stoch_signal()
                    
                elif name == 'obv':
                    obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
                    df['obv'] = obv.on_balance_volume()
                    
                elif name == 'mfi':
                    mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
                    df[f'mfi_{window}'] = mfi.money_flow_index()
            except Exception as e:
                logger.error(f"Error calculating {name} indicator: {str(e)}")
                raise
        
        logger.info("Technical indicators calculation completed")
        return df

    def fit_transform(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw price data and calculate technical indicators.
        
        Expected columns in price_df:
            - date (datetime)
            - open (float)
            - high (float)
            - low (float)
            - close (float)
            - volume (float)
            
        Returns:
            DataFrame with calculated features
        """
        logger.info("Starting price data processing...")
        
        # Validate input
        self._validate_input(price_df)
        
        # Calculate technical indicators
        price_df = self._calculate_technical_indicators(price_df)
        
        logger.info(f"Price data processing completed. Final shape: {price_df.shape}")
        return price_df

def run_price_pipeline(price_df: pd.DataFrame):
    """
    End-to-end pipeline for price data processing.
    
    Args:
        price_df (pd.DataFrame): Raw price data
        
    Returns:
        pd.DataFrame: Processed price data with additional features
    """
    logger.info("Starting price pipeline...")
    pipeline = PricePipeline()
    processed_df = pipeline.fit_transform(price_df)
    logger.info("Price pipeline completed successfully")
    return processed_df

def test():
    """
    Test function to demonstrate pipeline usage.
    """
    # Load sample price data from CSV
    logger.info("Loading sample price data...")
    df = pd.read_csv('data/raw/snp500/snp500.csv')
    
    # Run pipeline
    processed_df = run_price_pipeline(df)
    
    logger.info("\nProcessed DataFrame Info:")
    logger.info(f"Shape: {processed_df.shape}")
    logger.info("\nColumns:")
    logger.info(processed_df.columns.tolist())
    logger.info("\nFirst few rows:")
    logger.info(processed_df.head())
    
    # Save processed DataFrame to CSV
    output_path = 'data/pipeline/processed_price_data.csv'
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Processed DataFrame saved to {output_path}")

if __name__ == "__main__":
    test()
