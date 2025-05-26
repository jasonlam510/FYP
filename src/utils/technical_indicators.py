"""
Technical Indicators Utility Module

This module provides functions for calculating various technical indicators
using the ta (Technical Analysis) library.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import CCIIndicator
from typing import List, Dict
from logger import setup_logging

logger = setup_logging(__name__)

def calculate_technical_indicators(df: pd.DataFrame, indicators: List[Dict]) -> pd.DataFrame:
    """
    Calculate technical indicators for the given price data using ta library.
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
        indicators (List[Dict]): List of indicator configurations
            Each dict should have:
            - name: str, name of the indicator
            - window: int (optional), window size for the indicator
            
    Returns:
        pd.DataFrame: DataFrame with calculated indicators
        
    Example:
        indicators = [
            {'name': 'bb', 'window': 20},
            {'name': 'rsi', 'window': 14}
        ]
        df_with_indicators = calculate_technical_indicators(price_df, indicators)
    """
    if df is None:
        raise ValueError("DataFrame is not initialized")
        
    logger.info("Calculating technical indicators...")
    
    for indicator in indicators:
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

def test_technical_indicators():
    """
    Test function to demonstrate technical indicators calculation.
    Uses S&P 500 daily data and calculates all specified indicators.
    """
    # Define indicators to calculate
    INDICATORS = [
        {'name': 'bb', 'window': 20},
        {'name': 'ma', 'window': 20},
        {'name': 'ema', 'window': 20},
        {'name': 'rsi', 'window': 14},
        {'name': 'macd', 'window': 26},  # MACD slow window
        {'name': 'atr', 'window': 14},
        {'name': 'cci', 'window': 20},
        {'name': 'stochastic', 'window': 14},
        {'name': 'obv'},
        {'name': 'mfi', 'window': 14}
    ]
    
    # Read the data
    input_path = "data/raw/yf/SPX_1d.csv"
    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Print initial state
    logger.info("Initial DataFrame head:")
    print("\nInitial DataFrame head:")
    print(df.head())
    
    # Calculate indicators
    logger.info("Calculating technical indicators...")
    df = calculate_technical_indicators(df, INDICATORS)
    
    # Print results
    logger.info("Processed DataFrame head:")
    print("\nProcessed DataFrame head:")
    print(df.head())
    
    # Print summary statistics
    logger.info("Summary statistics of generated indicators:")
    print("\nSummary statistics:")
    print(df.describe())
    
    # Print list of all calculated indicators
    logger.info("List of calculated indicators:")
    print("\nCalculated indicators:")
    indicator_columns = [col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    print(indicator_columns)
    
    return df

def create_data_with_indicators():
    """
    Create a dataframe with technical indicators and price data.
    """
    INDICATORS = [
        {'name': 'bb', 'window': 20},
        {'name': 'ma', 'window': 20},
        {'name': 'ema', 'window': 20},
        {'name': 'rsi', 'window': 14},
        {'name': 'macd', 'window': 26},  # MACD slow window
        {'name': 'atr', 'window': 14},
        {'name': 'cci', 'window': 20},
        {'name': 'stochastic', 'window': 14},
        {'name': 'obv'},
        {'name': 'mfi', 'window': 14}
    ]
    # Read the data
    input_path = "data/raw/yf/SPX_1d.csv"
    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)

    # Calculate indicators
    logger.info("Calculating technical indicators...")
    df = calculate_technical_indicators(df, INDICATORS)

    # Save the data
    output_path = "data/processed/yf/SPX_1d_with_indicators.csv"
    logger.info(f"Saving data to {output_path}")
    df.to_csv(output_path, index=False)
    
    return df


if __name__ == '__main__':
    create_data_with_indicators() 