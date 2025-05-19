import pandas as pd
from ta.utils import dropna
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

def add_indicators(df, indicators):
    """Function to add multiple technical indicators to the DataFrame."""
    
    for indicator in indicators:
        indicator_lower = indicator['name'].lower()
        window = indicator.get('window', 20)  # Default window size is 20 if not specified
        
        if indicator_lower == 'bb':
            if f'bb_{window}' not in df.columns:
                # Initialize Bollinger Bands Indicator
                indicator_bb = BollingerBands(close=df["Close"], window=window, window_dev=2)
                # Add Bollinger Bands features with new column names
                df[f'bb_bbm_{window}'] = indicator_bb.bollinger_mavg()
                df[f'bb_bbh_{window}'] = indicator_bb.bollinger_hband()
                df[f'bb_bbl_{window}'] = indicator_bb.bollinger_lband()
                df[f'bb_bbhi_{window}'] = indicator_bb.bollinger_hband_indicator()
                df[f'bb_bbli_{window}'] = indicator_bb.bollinger_lband_indicator()
                df[f'bb_bbw_{window}'] = indicator_bb.bollinger_wband()
                df[f'bb_bbp_{window}'] = indicator_bb.bollinger_pband()
        
        elif indicator_lower == 'ma':
            if f'ma_{window}' not in df.columns:
                df[f'ma_{window}'] = SMAIndicator(close=df['Close'], window=window).sma_indicator()
        
        elif indicator_lower == 'ema':
            if f'ema_{window}' not in df.columns:
                df[f'ema_{window}'] = EMAIndicator(close=df['Close'], window=window).ema_indicator()
        
        elif indicator_lower == 'rsi':
            if f'rsi_{window}' not in df.columns:
                df[f'rsi_{window}'] = RSIIndicator(close=df['Close'], window=window).rsi()
        
        elif indicator_lower == 'macd':
            if f'macd_{window}' not in df.columns:
                macd = MACD(close=df['Close'], window_slow=window, window_fast=12, window_sign=9)
                df[f'macd_{window}'] = macd.macd()
        
        elif indicator_lower == 'atr':
            if f'atr_{window}' not in df.columns:
                atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)
                df[f'atr_{window}'] = atr.average_true_range()
        
        elif indicator_lower == 'cci':
            if f'cci_{window}' not in df.columns:
                df[f'cci_{window}'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window).cci()
        
        elif indicator_lower == 'stochastic':
            if f'stoch_{window}' not in df.columns:
                stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
                df[f'stoch_{window}'] = stoch.stoch()
        
        elif indicator_lower == 'obv':
            if 'obv' not in df.columns:
                df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        
        elif indicator_lower == 'mfi':
            if f'mfi_{window}' not in df.columns:
                df[f'mfi_{window}'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window).money_flow_index()

def example():
    # Load data
    df = pd.read_csv('data/raw/snp500/snp500.csv', sep=',')
    # Clean NaN values
    df = dropna(df)

    # Add indicators with specified windows
    # 1-day prediction indicator windows (short/standard)
    one_day_pred_indicators = [
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

    # 3-day prediction indicator windows (mix of short and medium)
    three_day_pred_indicators = [
        {'name': 'bb', 'window': 20},
        {'name': 'bb', 'window': 50},
        {'name': 'ma', 'window': 10},
        {'name': 'ma', 'window': 20},
        {'name': 'ma', 'window': 50},
        {'name': 'ema', 'window': 10},
        {'name': 'ema', 'window': 20},
        {'name': 'ema', 'window': 50},
        {'name': 'rsi', 'window': 14},
        {'name': 'rsi', 'window': 21},
        {'name': 'macd', 'window': 26},
        {'name': 'atr', 'window': 14},
        {'name': 'atr', 'window': 21},
        {'name': 'cci', 'window': 20},
        {'name': 'cci', 'window': 50},
        {'name': 'stochastic', 'window': 14},
        {'name': 'obv'},
        {'name': 'mfi', 'window': 14},
        {'name': 'mfi', 'window': 21}
    ]

    # 5-day prediction indicator windows (medium and long term)
    five_day_pred_indicators = [
        {'name': 'bb', 'window': 20},
        {'name': 'bb', 'window': 50},
        {'name': 'ma', 'window': 20},
        {'name': 'ma', 'window': 50},
        {'name': 'ma', 'window': 100},
        {'name': 'ema', 'window': 20},
        {'name': 'ema', 'window': 50},
        {'name': 'ema', 'window': 100},
        {'name': 'rsi', 'window': 14},
        {'name': 'rsi', 'window': 21},
        {'name': 'rsi', 'window': 28},
        {'name': 'macd', 'window': 26},
        {'name': 'atr', 'window': 14},
        {'name': 'atr', 'window': 21},
        {'name': 'atr', 'window': 28},
        {'name': 'cci', 'window': 20},
        {'name': 'cci', 'window': 50},
        {'name': 'stochastic', 'window': 14},
        {'name': 'obv'},
        {'name': 'mfi', 'window': 14},
        {'name': 'mfi', 'window': 21},
        {'name': 'mfi', 'window': 28}
    ]

    add_indicators(df, one_day_pred_indicators)

    # Save the dataframe to a new CSV file
    df.to_csv('data/processed/snp500/snp500.csv', index=False)

if __name__ == "__main__":
    example()   
