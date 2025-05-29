import pandas as pd
import numpy as np

def aggregate_news_price_rolling(
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

def combine_mi_price(mi_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine macro indicators with price data.
    
    Args:
        mi_df: DataFrame with macro indicators (from FRED), date as unnamed index
        price_df: DataFrame with price data, has date column with time
    
    Returns:
        Combined DataFrame with price data and macro indicators
    """
    # Create a copy of price_df
    combined_df = price_df.copy()
    
    # Reset index of mi_df and rename to 'date'
    mi_df = mi_df.reset_index()
    mi_df = mi_df.rename(columns={'index': 'date'})
    
    # Convert mi_df date to UTC and normalize to midnight
    mi_df['date'] = pd.to_datetime(mi_df['date'], utc=True).dt.normalize()
    
    # Set date as index for both dataframes
    combined_df.set_index('date', inplace=True)
    mi_df.set_index('date', inplace=True)
    
    # Reindex mi_df to match price_df's index and forward fill values
    mi_df = mi_df.reindex(combined_df.index, method='ffill')
    
    # Concatenate the dataframes
    result = pd.concat([combined_df, mi_df], axis=1)
    
    # Reset index to get date back as a column
    result = result.reset_index()
    
    return result