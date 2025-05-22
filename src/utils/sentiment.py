from transformers import pipeline
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np

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

# Function to get sentiment scores and add them to the DataFrame
def add_sentiment_scores(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # Create empty list to hold the final sentiment scores
    sentiment_scores = []
    
    # Use tqdm to track the progress of the loop
    for text in tqdm(df[column_name], desc="Processing Sentiment Scores", unit="text"):
        result = pipe(text)
        
        # Extract scores for positive, neutral, and negative sentiments
        positive_score = next(item['score'] for item in result[0] if item['label'] == 'positive')
        neutral_score = next(item['score'] for item in result[0] if item['label'] == 'neutral')
        negative_score = next(item['score'] for item in result[0] if item['label'] == 'negative')
        
        # Calculate and append the combined sentiment score
        sentiment_score = sentiment_to_score(positive_score, neutral_score, negative_score)
        sentiment_scores.append(sentiment_score)
    
    # Add only the final sentiment score to the DataFrame
    df['sentiment_score_finbert'] = sentiment_scores
    
    return df