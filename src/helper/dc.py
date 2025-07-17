import pandas as pd
import numpy as np

def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    """Calculate the rolling Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _detect_atr_dc_events(df: pd.DataFrame, atr_window: int = 10, k: float = 1.5) -> pd.DataFrame:
    """
    Detect ATR-based directional-change events.
    Returns a DataFrame with columns:
      'event_date', 'event_type', 'event_price', 'threshold', 'atr'
    """
    atr = _compute_atr(df['high'], df['low'], df['close'], atr_window)
    mode = None
    last_extremum_price = df['close'].iloc[0]
    events = []

    for i in range(1, len(df)):
        date = df.index[i]
        price = df['close'].iloc[i]
        thresh = k * atr.iloc[i]

        if mode in (None, 'down') and price >= last_extremum_price + thresh:
            mode = 'up'
            last_extremum_price = price
            events.append((date, 'up', price, thresh, atr.iloc[i]))
        elif mode in (None, 'up') and price <= last_extremum_price - thresh:
            mode = 'down'
            last_extremum_price = price
            events.append((date, 'down', price, thresh, atr.iloc[i]))
        else:
            # update extremum within trend
            if mode == 'up':
                last_extremum_price = max(last_extremum_price, price)
            elif mode == 'down':
                last_extremum_price = min(last_extremum_price, price)

    return pd.DataFrame(events, columns=['event_date', 'event_type', 'event_price', 'threshold', 'atr'])


def add_dc_event_features(df: pd.DataFrame,
                       atr_window: int = 10,
                       k: float = 1.5,
                       rv_window: int = 10,
                       momentum_window: int = 5,
                       vol_ma_window: int = 5) -> pd.DataFrame:
    """
    Augment price_df with event-based and contextual features:
      - overshoot_ratio
      - time_since_last_dc
      - return_since_last_dc
      - atr_at_event
      - realized_volatility
      - momentum
      - volume_spike_ratio
    """
    # Create a copy and set index without modifying original
    price_df = df.copy()
    price_df = price_df.set_index('date')

    # 1) Detect events
    events = _detect_atr_dc_events(price_df, atr_window, k)

    # 2) Compute overshoot for each event
    overshoots = []
    for idx in range(len(events)-1):
        start = events.loc[idx, 'event_date']
        end = events.loc[idx+1, 'event_date']
        window = price_df.loc[start:end]
        if events.loc[idx, 'event_type'] == 'up':
            extremum = window['high'].max()
            overshoot = (extremum - events.loc[idx, 'event_price']) / events.loc[idx, 'threshold']
        else:
            extremum = window['low'].min()
            overshoot = (events.loc[idx, 'event_price'] - extremum) / events.loc[idx, 'threshold']
        overshoots.append(overshoot)
    overshoots.append(np.nan)  # no overshoot for last event
    events['overshoot_ratio'] = overshoots

    # 3) Compute time and return since last DC
    events['time_since_last_dc'] = (events['event_date'] - events['event_date'].shift(1)).dt.days
    events['return_since_last_dc'] = events['event_price'].pct_change()

    # 4) Map event features back to daily DataFrame
    for feat in ['overshoot_ratio', 'time_since_last_dc', 'return_since_last_dc', 'atr']:
        price_df[feat] = events.set_index('event_date')[feat].reindex(price_df.index, method='ffill')

    # 5) Realized volatility: rolling std of daily returns
    price_df['realized_volatility'] = price_df['close'].pct_change().rolling(rv_window).std()

    # 6) Momentum: price difference over window
    price_df['momentum'] = price_df['close'] - price_df['close'].shift(momentum_window)

    # 7) Volume spike ratio
    price_df['vol_ma'] = price_df['volume'].rolling(vol_ma_window).mean()
    price_df['volume_spike_ratio'] = price_df['volume'] / price_df['vol_ma']

    # clean up helper column
    price_df = price_df.drop(columns=['vol_ma'])

    # Reset index and ensure date column is preserved
    price_df = price_df.reset_index()
    
    # Ensure the date column is in the same format as the input
    price_df['date'] = pd.to_datetime(price_df['date'])
    
    return price_df
