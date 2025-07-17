import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, VolumeWeightedAveragePrice
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator, ADXIndicator, VortexIndicator
from typing import List, Dict

def calculate_technical_indicators(price_df: pd.DataFrame, indicators: List[Dict]) -> pd.DataFrame:
    """
    Calculate technical indicators for the given price data using ta library.
    
    Args:
        price_df (pd.DataFrame): Price data with OHLCV columns
        indicators (List[Dict]): List of indicator configurations
        
    Returns:
        pd.DataFrame: DataFrame with calculated indicators
    """
    if price_df is None:
        raise ValueError("DataFrame is not initialized")
    
    df = price_df.copy()
    
    for indicator in indicators:
        name = indicator['name']
        window = indicator.get('window', None)
        
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
                
            elif name == 'adx':
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
                df[f'adx_{window}'] = adx.adx()
                df[f'di_pos_{window}'] = adx.adx_pos()
                df[f'di_neg_{window}'] = adx.adx_neg()
                
            elif name == 'vortex':
                vortex = VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
                df[f'vortex_pos_{window}'] = vortex.vortex_indicator_pos()
                df[f'vortex_neg_{window}'] = vortex.vortex_indicator_neg()
                
            elif name == 'obv':
                obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
                df['obv'] = obv.on_balance_volume()
                
            elif name == 'mfi':
                mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
                df[f'mfi_{window}'] = mfi.money_flow_index()
                
            elif name == 'vwap':
                vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
                df['vwap'] = vwap.volume_weighted_average_price()

            elif name == 'roc':
                roc = ROCIndicator(close=df['close'], window=window)
                df[f'roc_{window}'] = roc.roc()

            else:
                raise ValueError(f"Unknown indicator: {name}")
                
        except Exception as e:
            raise e
            # Dont continue with next indicator if one fails
    
    return df