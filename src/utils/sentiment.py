from transformers import pipeline
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np
import asyncio
from logger import setup_logging

logger = setup_logging(__name__)

# Initialize the sentiment analysis pipeline with FinBERT
pipe = pipeline("text-classification", model="prosusai/finbert", top_k=None)

def sentiment_to_score(positive: float, neutral: float, negative: float) -> float:
    """
    Convert sentiment scores to a single value between -1 and 1 using softmax.
    
    Args:
        positive (float): Positive sentiment score
        neutral (float): Neutral sentiment score
        negative (float): Negative sentiment score
        
    Returns:
        float: A single sentiment score between -1 and 1
    """
    # Apply softmax to get probabilities
    scores = np.array([positive, neutral, negative])
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probabilities = exp_scores / exp_scores.sum()
    
    # Convert to -1 to 1 scale
    # We weight positive as 1, neutral as 0, and negative as -1
    weighted_score = probabilities[0] * 1 + probabilities[1] * 0 + probabilities[2] * -1
    
    return weighted_score

async def add_sentiment_scores(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Add sentiment scores to the DataFrame using FinBERT model.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing text data
        column_name (str): Name of the column containing text to analyze
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment scores
    """
    logger.info("Starting sentiment analysis...")
    
    # Create empty list to hold the final sentiment scores
    sentiment_scores = []
    
    # Process texts in batches to allow for async operation
    batch_size = 10
    for i in range(0, len(df), batch_size):
        batch_texts = df[column_name].iloc[i:i+batch_size]
        
        # Process batch
        for text in tqdm(batch_texts, desc=f"Processing batch {i//batch_size + 1}", unit="text"):
            # Run sentiment analysis in a thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(None, pipe, text)
            
            # Extract scores for positive, neutral, and negative sentiments
            positive_score = next(item['score'] for item in result[0] if item['label'] == 'positive')
            neutral_score = next(item['score'] for item in result[0] if item['label'] == 'neutral')
            negative_score = next(item['score'] for item in result[0] if item['label'] == 'negative')
            
            # Calculate and append the combined sentiment score
            sentiment_score = sentiment_to_score(positive_score, neutral_score, negative_score)
            sentiment_scores.append(sentiment_score)
    
    # Add only the final sentiment score to the DataFrame
    df['sentiment_score_finbert'] = sentiment_scores
    
    logger.info("Sentiment analysis completed")
    return df

async def test_sentiment_analysis():
    """
    Test function to demonstrate the sentiment analysis functionality.
    
    This function creates a sample DataFrame with financial news headlines
    and demonstrates how to use the sentiment analysis.
    
    Example:
        >>> import asyncio
        >>> asyncio.run(test_sentiment_analysis())
    """
    # Sample financial news headlines
    sample_data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'title': [
            'Apple Stock Surges to New High After Strong Earnings Report',
            'Market Uncertainty Grows as Fed Signals Potential Rate Hike',
            'Tesla Shares Drop Following Production Delay Announcement'
        ],
        'content': [
            'Apple Inc. reported record-breaking quarterly earnings, exceeding market expectations. The tech giant\'s stock price soared to new heights as investors responded positively to the strong financial performance.',
            'Federal Reserve officials have indicated a possible interest rate increase in the coming months, creating uncertainty in the financial markets. Analysts are divided on the potential impact.',
            'Tesla announced unexpected delays in its production schedule, causing concern among investors. The electric vehicle manufacturer\'s stock price declined significantly following the news.'
        ],
        'source': ['Financial Times', 'Wall Street Journal', 'Bloomberg'],
        'url': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3']
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    logger.info("Testing sentiment analysis with sample data...")
    logger.info("\nSample data:")
    logger.info(df[['title', 'content']].to_string())
    
    # Add sentiment scores
    df_with_sentiment = await add_sentiment_scores(df, 'content')
    
    # Display results
    logger.info("\nResults with sentiment scores:")
    logger.info(df_with_sentiment[['title', 'sentiment_score_finbert']].to_string())
    
    # Print summary statistics
    logger.info("\nSentiment Score Statistics:")
    logger.info(f"Mean sentiment: {df_with_sentiment['sentiment_score_finbert'].mean():.3f}")
    logger.info(f"Min sentiment: {df_with_sentiment['sentiment_score_finbert'].min():.3f}")
    logger.info(f"Max sentiment: {df_with_sentiment['sentiment_score_finbert'].max():.3f}")
    
    return df_with_sentiment

if __name__ == "__main__":
    asyncio.run(test_sentiment_analysis())