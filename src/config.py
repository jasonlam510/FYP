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
N_TRIALS = 1
HALF_LIFE_RANGE = (0.5, 6)
N_DAYS_RANGE = (1, 3)

# Technical Indicators for Price Data
BALANCED_INDICATORS = [
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
]

MULTI_TIMEFRAME_INDICATORS = [
    {'name': 'ema',        'window': 5},    # fast EMA for very short-term momentum
    {'name': 'roc',        'window': 5},    # 5-period Rate of Change for quick swings
    {'name': 'rsi',        'window': 14},   # Relative Strength Index for short-term momentum
    {'name': 'stochastic', 'window': 14},   # Stochastic oscillator (%K)
    {'name': 'atr',        'window': 14},   # Average True Range for short-term volatility
    {'name': 'ma',         'window': 20},   # 20-period SMA for mid-term trend
    {'name': 'bb',         'window': 20},   # 20-period Bollinger Bands
    {'name': 'macd',       'window': 26},   # MACD slow window (fast=12, signal=9)
    {'name': 'adx',        'window': 14},   # Average Directional Index for trend strength
    {'name': 'mfi',        'window': 14},   # Money Flow Index for volume-price interplay
    {'name': 'vortex',     'window': 14},   # Vortex Indicator for trend confirmation
    {'name': 'ma',         'window': 50},   # 50-period SMA for longer-term trend
    {'name': 'ema',        'window': 50},   # 50-period EMA
    {'name': 'ma',         'window': 200},  # 200-period SMA for very long-term trend
    {'name': 'cci',        'window': 50},   # Commodity Channel Index for cyclical deviations
    {'name': 'obv'},                       # On-Balance Volume (cumulative) for volume confirmation
] 