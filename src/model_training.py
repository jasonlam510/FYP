# model_training.ipynb
# # Intro
# This Notebook is to aggretate the dataset that is extract feature from finbert and LLM. 

# %% [markdown]
# # News Data

# %% [markdown]
# ## Data import

# %%
import pandas as pd

news_df = pd.read_csv('finbert_llm_sentiment_inserted.csv')

print(news_df.shape)
display(news_df.head())
news_df

# %% [markdown]
# ## Remove non model input features
# For better performance in weak computer(small amount of ram)

# %%
news_df = news_df.drop(columns=['headline'])
display(news_df.head())

# %% [markdown]
# ## Sort by Date

# %%
# sort by date
news_df = news_df.sort_values('date')
display(news_df.head())

# %% [markdown]
# # Price data

# %% [markdown]
# ## yFinance library

# %%
import yfinance as yf

def download_finance_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """Download financial data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (e.g., '^SPX')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d' for daily, '1h' for hourly, etc.)
    
    Returns:
        pd.DataFrame: The downloaded data
    """
    # Download data from Yahoo Finance
    print(f"Downloading {interval} data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Print data information
    print(f"\nData shape: {df.shape}")
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    return df

# %% [markdown]
# ## Download the price data by the date range of the news data

# %%
# Convert date column to datetime first
news_df['date'] = pd.to_datetime(news_df['date'])

# Get the min and max dates from news_df (day only)
start_date = news_df['date'].min().strftime('%Y-%m-%d')
end_date = news_df['date'].max().strftime('%Y-%m-%d')

print(f"Date range: {start_date} to {end_date}")

# %%
# Download S&P 500 data for the date range
spx_df = None
try:
    spx_df = download_finance_data(
        symbol='^SPX',
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )
    # Display first few rows
    print("\nFirst few rows of S&P 500 data:")
    print(spx_df.head())
except Exception as e:
    print(f"Error downloading data: {e}")
    pass


# %% [markdown]
# Note: When working with yFinance API, you may encounter common issues such as rate limiting or discontinued symbols. 
# These are typical challenges when working with financial data APIs. If you experience these issues:
# 1. First try updating to the latest version of yFinance using: pip install --upgrade yfinance
# 2. If problems persist, you can manually download the data from Yahoo Finance and import it using the code below

# %%
if spx_df is None:
    spx_df = pd.read_csv('SPX_1d.csv')
spx_df.head()

# %% [markdown]
# 

# %% [markdown]
# ## Insert Technical Indicators

# %%
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import CCIIndicator
from typing import List, Dict

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
                
            elif name == 'obv':
                obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
                df['obv'] = obv.on_balance_volume()
                
            elif name == 'mfi':
                mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=window)
                df[f'mfi_{window}'] = mfi.money_flow_index()
        except Exception as e:
            raise Exception(f"Error calculating {name} indicator: {str(e)}")
    
    return df

# %% [markdown]
# ## Indicators List

# %%
# Convert the column name to lower case
spx_df.columns = spx_df.columns.str.lower()

# INDICATORS = [
#         {'name': 'bb', 'window': 20},
#         {'name': 'ma', 'window': 20},
#         {'name': 'ema', 'window': 20},
#         {'name': 'rsi', 'window': 14},
#         {'name': 'macd', 'window': 26},  # MACD slow window
#         {'name': 'atr', 'window': 14},
#         {'name': 'cci', 'window': 20},
#         {'name': 'stochastic', 'window': 14},
#         {'name': 'obv'},
#         {'name': 'mfi', 'window': 14}
#     ]
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
# INDICATORS = [
#     {'name': 'ema',        'window': 5},    # fast EMA for very short-term momentum :contentReference[oaicite:0]{index=0}
#     {'name': 'roc',        'window': 5},    # 5-period Rate of Change for quick swings
#     {'name': 'rsi',        'window': 14},   # Relative Strength Index for short-term momentum :contentReference[oaicite:1]{index=1}
#     {'name': 'stochastic', 'window': 14},   # Stochastic oscillator (%K) :contentReference[oaicite:2]{index=2}
#     {'name': 'atr',        'window': 14},   # Average True Range for short-term volatility :contentReference[oaicite:3]{index=3}
#     {'name': 'ma',         'window': 20},   # 20-period SMA for mid-term trend
#     {'name': 'bb',         'window': 20},   # 20-period Bollinger Bands :contentReference[oaicite:4]{index=4}
#     {'name': 'macd',       'window': 26},   # MACD slow window (fast=12, signal=9) :contentReference[oaicite:5]{index=5}
#     {'name': 'adx',        'window': 14},   # Average Directional Index for trend strength :contentReference[oaicite:6]{index=6}
#     {'name': 'mfi',        'window': 14},   # Money Flow Index for volume-price interplay :contentReference[oaicite:7]{index=7}
#     {'name': 'vortex',     'window': 14},   # Vortex Indicator for trend confirmation
#     {'name': 'ma',         'window': 50},   # 50-period SMA for longer-term trend
#     {'name': 'ema',        'window': 50},   # 50-period EMA :contentReference[oaicite:8]{index=8}
#     {'name': 'ma',         'window': 200},  # 200-period SMA for very long-term trend :contentReference[oaicite:9]{index=9}
#     {'name': 'cci',        'window': 50},   # Commodity Channel Index for cyclical deviations
#     {'name': 'obv'},                       # On-Balance Volume (cumulative) for volume confirmation
#     {'name': 'vwap'}                       # Volume-Weighted Average Price for intraday anchoring
# ]

# spx_df = calculate_technical_indicators(spx_df, INDICATORS)

# %% [markdown]
# # DC Indicator

# %%
import pandas as pd
import numpy as np

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    """Calculate the rolling Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def detect_atr_dc_events(df: pd.DataFrame, atr_window: int = 10, k: float = 1.5) -> pd.DataFrame:
    """
    Detect ATR-based directional-change events.
    Returns a DataFrame with columns:
      'event_date', 'event_type', 'event_price', 'threshold', 'atr'
    """
    atr = compute_atr(df['high'], df['low'], df['close'], atr_window)
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


def add_event_features(price_df: pd.DataFrame,
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
    df = price_df.copy()
    df.set_index('date', inplace=True)

    # 1) Detect events
    events = detect_atr_dc_events(df, atr_window, k)

    # 2) Compute overshoot for each event
    overshoots = []
    for idx in range(len(events)-1):
        start = events.loc[idx, 'event_date']
        end = events.loc[idx+1, 'event_date']
        window = df.loc[start:end]
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
        df[feat] = events.set_index('event_date')[feat].reindex(df.index, method='ffill')

    # 5) Realized volatility: rolling std of daily returns
    df['realized_volatility'] = df['close'].pct_change().rolling(rv_window).std()

    # 6) Momentum: price difference over window
    df['momentum'] = df['close'] - df['close'].shift(momentum_window)

    # 7) Volume spike ratio
    df['vol_ma'] = df['volume'].rolling(vol_ma_window).mean()
    df['volume_spike_ratio'] = df['volume'] / df['vol_ma']

    # clean up helper column
    df.drop(columns=['vol_ma'], inplace=True)

    df.reset_index(inplace=True)
    return df


# %%
# # Convert to UTC
# spx_df['date'] = pd.to_datetime(spx_df['date'], utc=True)
# spx_df['date'] = spx_df['date'].dt.tz_convert('UTC')

# spx_df = add_event_features(spx_df, atr_window=10, k=2.0)
# spx_df.head()

# # Sort news dataframe by date
# news_df = news_df.sort_values('date')

# # No need to convert timezone since it's already in US/Eastern
# # Just ensure it's datetime
# news_df['date'] = pd.to_datetime(news_df['date'], utc=True)

# %% [markdown]
# # Aggreate Data

# %% [markdown]
# ## Event List

# %%
EVENT_TYPES = [
    "earnings", "merger", "dividend", "guidance", "regulatory",
    "macroeconomic", "monetary", "CEO change", "product launch",
    "supply-chain", "credit", "scandal", "analyst", "sector-wide",
    "geopolitical", "other"
]

# %% [markdown]
# ## Aggreate by rolling window

# %%
import pandas as pd
import numpy as np

def aggregate_news_rolling(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    event_types: list,
    n_days: int = 3,
    half_life_days: float = 1.5
):
    """
    For each trading day, aggregate news from the previous n_days (including the current day).
    Args:
        news_df: DataFrame with ['date', 'relevance_score', 'event_importance', 'event_type', 'sentiment_score_llm']
        price_df: DataFrame with ['date', 'close', 'volume']
        event_types: List of all possible event types
        n_days: Number of days in the rolling window
        half_life_days: Half-life for exponential decay (in days)
    Returns:
        pd.DataFrame: Aggregated data with one-hot encoded event types and price data
    """
    # Ensure datetime
    news_df = news_df.copy()
    price_df = price_df.copy()
    news_df['date'] = pd.to_datetime(news_df['date'], utc=True)
    price_df['date'] = pd.to_datetime(price_df['date'], utc=True)

    # Prepare price data - convert dates to 4 PM ET trading days
    price_df['trading_day'] = price_df['date'].apply(
        lambda x: pd.Timestamp(
            year=x.year,
            month=x.month,
            day=x.day,
            hour=16,
            minute=0,
            second=0,
            tz='US/Eastern'
        )
    )

    # Prepare output list
    agg_list = []

    for trading_day in price_df['trading_day']:
        # Define window: n_days before (inclusive)
        window_start = trading_day - pd.Timedelta(days=n_days-1)
        window_end = trading_day

        # Select news in the window
        mask = (news_df['date'] >= window_start) & (news_df['date'] <= window_end)
        news_window = news_df.loc[mask].copy()

        # Compute decay factor for each news item (relative to trading day)
        lam = np.log(2) / half_life_days
        delta = (trading_day - news_window['date']).dt.total_seconds() / (24*3600)
        delta = np.clip(delta, 0, None)
        news_window['decay'] = np.exp(-lam * delta)

        # Calculate weighted sentiment score
        news_window['weighted_sentiment_llm'] = (
            news_window['sentiment_score_llm'] *
            news_window['relevance_score'] *
            news_window['event_importance'] *
            news_window['decay']
        )

        # One-hot encode event_type
        news_window['event_type'] = pd.Categorical(news_window['event_type'], categories=event_types)
        dummies = pd.get_dummies(news_window['event_type']).reindex(columns=event_types, fill_value=0)
        weighted_dummies = dummies.mul(news_window['weighted_sentiment_llm'], axis=0)

        # Aggregate by sum for this trading day
        agg_row = weighted_dummies.sum(axis=0)
        agg_row['trading_day'] = trading_day

        agg_list.append(agg_row)

    # Combine all rows
    news_agg = pd.DataFrame(agg_list).fillna(0.0)

    # Merge with price data
    merged_df = pd.merge(
        price_df,
        news_agg,
        on='trading_day',
        how='left'
    ).fillna(0.0)

    # Sort by trading day
    merged_df = merged_df.sort_values('trading_day').reset_index(drop=True)

    return merged_df

# %%
spx_df = calculate_technical_indicators(spx_df, INDICATORS)
aggregated_df = aggregate_news_rolling(
    news_df=news_df,
    price_df=spx_df,
    event_types=EVENT_TYPES,
    n_days=3,                # Use a 3-day rolling window
    half_life_days=2,
)

display(aggregated_df.head())
aggregated_df


# %% [markdown]
# # Save csv

# %%
aggregated_df.to_csv('aggregated_data.csv', index=False)

# %% [markdown]
# # Model Training

# %% [markdown]
# ## Training Pipeline

# %% [markdown]
# ### Import Library

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, f1_score


def create_sequences(data, seq_length, target_col_idx):
    """Create sequences for time series data."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)

def prepare_data_for_model(df: pd.DataFrame, seq_length: int, test_size: float = 0.2) -> Tuple:
    """Prepare data for the model, handling datetime columns properly."""
    # Create a copy of the dataframe
    df = df.copy()
    
    # Store the datetime column separately
    datetime_col = df['date']
    
    # Get numeric columns (excluding datetime)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Scale the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Find the index of the 'close' column
    close_col_idx = list(numeric_cols).index('close')
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length, close_col_idx)
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Get corresponding dates for test set
    test_dates = datetime_col[train_size + seq_length:].values
    
    return X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols

# %% [markdown]
# ### LSTM+CNN

# %%
def build_cnn_lstm_model(trial, seq_length, n_features):
    """CNN+LSTM model with variable number of LSTM layers."""

    # CNN parameters
    cnn_filters = trial.suggest_int('cnn_filters', 32, 128)
    cnn_kernel = trial.suggest_int('cnn_kernel', 2, 5)
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.1, 0.5)
    
    # LSTM parameters
    lstm_layers = trial.suggest_int('lstm_layers', 2, 5)  # Let Optuna choose 1, 2, or 3 layers
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))
    model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(cnn_dropout))
    
    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        # Only the last LSTM layer should not return sequences
        return_seq = (i < lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_seq))
        model.add(Dropout(lstm_dropout))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# %% [markdown]
# ### LSTM

# %%
def build_lstm_model(trial, seq_length, n_features, **kwargs):
    """LSTM model with variable number of LSTM layers."""

    # Hyperparameters from best trial
    lstm_layers = 5
    lstm_units = 67
    dropout_rate = 0.1294676223336333
    learning_rate = 0.005698251072263444

    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        units = lstm_units if i == 0 else lstm_units // 2  # Optional: halve units for deeper layers
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# %% [markdown]
# ### Train and Evaluate

# %%
from typing import Union

def train_and_evaluate_model(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    event_types: list,
    seq_length: int,
    n_trials: int = 20,
    build_model_fn: Callable = build_lstm_model,
    model_params: Dict[str, Any] = None,
    half_life_range: Union[tuple, float] = (0.5, 10.0),
    n_days_range: Union[tuple, int] = (1, 10)
) -> Tuple:
    """
    Train and evaluate the model with hyperparameter optimization, including half_life_days as a hyperparameter.

    Args:
        news_df: Raw news DataFrame
        price_df: Raw price DataFrame
        event_types: List of event types for one-hot encoding
        seq_length: Length of input sequences
        n_trials: Number of Optuna trials
        build_model_fn: Model building function
        model_params: Additional model parameters
        half_life_range: (min, max) range for half_life_days

    Returns:
        Tuple containing (model, history, predictions, actual, test_dates, best_params)
    """

    def objective(trial):
        # 1. Get half_life_days and n_days for this trial
        half_life_days = half_life_range if isinstance(half_life_range, float) else trial.suggest_float('half_life_days', half_life_range[0], half_life_range[1])
        n_days = n_days_range if isinstance(n_days_range, int) else trial.suggest_int('n_days', n_days_range[0], n_days_range[1])

        # 2. Aggregate data using the suggested half_life_days
        aggregated_df = aggregate_news_rolling(
        news_df=news_df,
        price_df=spx_df,
        event_types=EVENT_TYPES,
        n_days=n_days,                
        half_life_days=half_life_days
        )

        # 3. Prepare data for the model
        X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
            aggregated_df, seq_length=seq_length
        )

        # 4. Split training data into train and validation sets
        val_size = int(len(X_train) * 0.2)
        X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

        # 5. Build and train the model
        model = build_model_fn(trial, seq_length, X_train.shape[2], **(model_params or {}))
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        # 6. Return the best validation loss
        return min(history.history['val_loss'])

    # 7. Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # 8. After finding the best trial, rebuild the dataset with the best half_life_days
    best_half_life = study.best_params.get('half_life_days', half_life_range if isinstance(half_life_range, float) else half_life_range[0])
    n_days = study.best_params.get('n_days', n_days_range if isinstance(n_days_range, int) else n_days_range[0])
    best_aggregated_df = aggregate_news_rolling(
        news_df, price_df, event_types, half_life_days=best_half_life, n_days=n_days
    )

    # 9. Prepare data for the final model
    X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
        best_aggregated_df, seq_length=seq_length
    )

    # 10. Train the final model with the best hyperparameters
    best_model = build_model_fn(study.best_trial, seq_length, X_train.shape[2], **(model_params or {}))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 11. Make predictions
    predictions = best_model.predict(X_test)
    close_col_idx = list(numeric_cols).index('close')
    pred_reshaped = np.zeros((len(predictions), len(numeric_cols)))
    pred_reshaped[:, close_col_idx] = predictions.flatten()
    predictions = scaler.inverse_transform(pred_reshaped)[:, close_col_idx]
    actual_reshaped = np.zeros((len(y_test), len(numeric_cols)))
    actual_reshaped[:, close_col_idx] = y_test
    actual = scaler.inverse_transform(actual_reshaped)[:, close_col_idx]

    # 11. Print the best hyperparameters found
    print("Best hyperparameters found:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"seq_length: {seq_length}")

    return best_model, history, predictions, actual, test_dates, study.best_params

# %% [markdown]
# ### Plot Results

# %%
def plot_results(predictions1, predictions2, actual, test_dates, history, model1_name="Model 1", model2_name="Model 2"):
    """Plot the results and training history."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot predictions vs actual
    ax1.plot(test_dates, actual, label='Actual', color='blue')
    ax1.plot(test_dates, predictions1, label=model1_name, color='red')
    ax1.plot(test_dates, predictions2, label=model2_name, color='green')
    ax1.set_title('Actual vs Predicted Stock Prices')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training history
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Calculate metrics for both models
    metrics = {
        'MSE': [],
        'Directional Accuracy': [],
        'F1 Score': []
    }
    
    for pred in [predictions1, predictions2]:
        # MSE
        mse = mean_squared_error(actual, pred)
        metrics['MSE'].append(mse)
        
        # Directional Accuracy
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(pred))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        metrics['Directional Accuracy'].append(directional_accuracy)
        
        # F1 Score
        actual_binary = (actual_direction > 0).astype(int)
        pred_binary = (pred_direction > 0).astype(int)
        f1 = f1_score(actual_binary, pred_binary)
        metrics['F1 Score'].append(f1)
    
    # Create comparison table
    table_data = [
        ['Metric', model1_name, model2_name],
        ['MSE', f'{metrics["MSE"][0]:.4f}', f'{metrics["MSE"][1]:.4f}'],
        ['Directional Accuracy', f'{metrics["Directional Accuracy"][0]:.4f}', f'{metrics["Directional Accuracy"][1]:.4f}'],
        ['F1 Score', f'{metrics["F1 Score"][0]:.4f}', f'{metrics["F1 Score"][1]:.4f}']
    ]
    
    # Add table to plot
    table = ax1.table(cellText=table_data,
                     loc='upper right',
                     cellLoc='center',
                     colWidths=[0.2, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.show()

# %%
seq_length = 10
n_trials = 20

# %% [markdown]
# # Test different Model

# %% [markdown]
# ## Balance Indicator Set

# %% [markdown]
# ### Test LSTM + CNN

# %%
best_model, history, LSTM_CNN_predictions, actual, test_dates, best_params = train_and_evaluate_model(
    news_df=news_df,
    price_df=spx_df,
    event_types=EVENT_TYPES,
    seq_length=seq_length,
    n_trials=n_trials,
    build_model_fn=build_cnn_lstm_model,  # or your custom model function
    model_params=None,                    # or pass a dict of extra model params if needed
    half_life_range=(0.5, 10.0),           # or any range you want to search
    n_days_range=(1, 10)
)

# Save the model to a file (e.g., HDF5 format)
best_model.save('LSTM_CNN_Model.keras')

# %% [markdown]
# ### Test LSTM

# %%
best_model, history, LSTM_predictions, actual, test_dates, best_params = train_and_evaluate_model(
    news_df=news_df,
    price_df=spx_df,
    event_types=EVENT_TYPES,
    seq_length=seq_length,
    n_trials=n_trials,
    build_model_fn=build_lstm_model,  # or your custom model function
    model_params=None,                    # or pass a dict of extra model params if needed
    half_life_range=(0.5, 10.0),           # or any range you want to search
    n_days_range=(1, 10)
)


# Save the model to a file (e.g., HDF5 format)
best_model.save('LSTM_Model.keras')


