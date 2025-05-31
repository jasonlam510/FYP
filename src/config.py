"""
Configuration file for the project.
Contains paths to data files and other constants.
"""

# Data file paths
DATA_DIR = "data"
NEWS_DATA_PATH = f"{DATA_DIR}/finbert_llm_sentiment_inserted_cleaned.csv"
PRICE_DATA_PATH = f"{DATA_DIR}/SPX_1d.csv"

# Event types for news classification
EVENT_TYPES = [
    "earnings", "merger", "dividend", "guidance", "regulatory",
    "macroeconomic", "monetary", "CEO change", "product launch",
    "supply-chain", "credit", "scandal", "analyst", "sector-wide",
    "geopolitical", "other"
]

# Model training parameters
SEQ_LENGTH = 10
N_TRIALS = 20
HALF_LIFE_RANGE = (0.5, 10.0)
N_DAYS_RANGE = (1, 10)

# Technical indicators configuration
INDICATORS = [
    {'name': 'bb', 'window': 20},          # Bollinger Bands
    {'name': 'ma', 'window': 50},          # Simple Moving Average
    {'name': 'ema', 'window': 12},         # Exponential Moving Average
    {'name': 'rsi', 'window': 14},         # Relative Strength Index
    {'name': 'macd', 'window': 26},        # MACD slow window (fast=12, signal=9)
    {'name': 'atr', 'window': 14},         # Average True Range
    {'name': 'cci', 'window': 20},         # Commodity Channel Index
    {'name': 'stochastic', 'window': 14},  # Stochastic oscillator (%K)
    {'name': 'adx', 'window': 14},         # Average Directional Index
    {'name': 'vortex', 'window': 14},      # Vortex Indicator
    {'name': 'obv'},                       # On-Balance Volume
    {'name': 'mfi', 'window': 14},         # Money Flow Index
    {'name': 'vwap'}                       # Volume-Weighted Average Price
] 