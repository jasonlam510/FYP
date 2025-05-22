import os
import pandas as pd
import json
import asyncio
from typing import List, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.llm.google_genai_client import generate_response
from src.utils.rate_limiter import RateLimiter
from src.utils.logger import setup_logging

# Configure logging
logger = setup_logging(__name__)

# Constants
MAX_CONCURRENT_TASKS = 500  # Match API's AFC limit
MAX_RPM = 4000  # Maximum requests per minute

# Initialize rate limiter
rate_limiter = RateLimiter(MAX_RPM, MAX_CONCURRENT_TASKS)

def build_prompt(content: str, content_name: str = "news", market_reference: str = "S&P 500") -> str:
    """
    Construct a generic prompt for Gemini based on arbitrary news content.
    """
    return (
        f"You are a financial‐news analysis assistant. Given only the {content_name} of the news, you must output a valid JSON object with exactly these four fields (no extra keys, no prose):\n"
        f"- sentiment_score: float between -1 (very negative) and 1 (very positive)  \n"
        f"- relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the {market_reference})  \n"
        "- event_importance: float between 0 (no market impact) and 1 (major market-moving event)  \n"
        "- event_type: one of [\"earnings\", \"merger\", \"dividend\", \"guidance\", \"regulatory\", \"macroeconomic\", \"monetary\", \"CEO change\", \"product launch\", \"supply-chain\", \"credit\", \"scandal\", \"analyst\", \"sector-wide\", \"geopolitical\", \"other\"]\n\n"
        f"**{content_name.capitalize()}:** \"{content}\"\n\n"
        "**Output only JSON.**"
    )

def parse_response(response_text: str) -> dict:
    """
    Parse the JSON content from the API response text.
    """
    cleaned = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nRaw response: {response_text}")
    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not an object: {data}")
    return data

async def process_row(row: pd.Series, idx: int, content_name: str, market_reference: str,
                     model: str, temperature: float, max_output_tokens: int, top_p: float, top_k: int) -> Dict[str, Any]:
    """
    Process a single row asynchronously.
    """
    content = row.get(content_name, '') or row.get(content_name.capitalize(), '')
    if not isinstance(content, str) or not content.strip():
        return {'idx': idx, 'error': 'Empty content'}

    prompt = build_prompt(content, content_name, market_reference)
    try:
        # Wait for rate limiter and active request slot
        await rate_limiter.acquire()
        await rate_limiter.start_request()
        
        try:
            resp_text = await generate_response(
                model=model,
                input_text=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k
            )
            parsed = parse_response(resp_text)
            return {
                'idx': idx,
                'sentiment_score': parsed.get('sentiment_score'),
                'relevance_score': parsed.get('relevance_score'),
                'event_importance': parsed.get('event_importance'),
                'event_type': parsed.get('event_type')
            }
        finally:
            await rate_limiter.end_request()
            
    except Exception as e:
        logger.error(f"Error processing row {idx} ('{content}'): {e}")
        return {'idx': idx, 'error': str(e)}

async def process_batch(df: pd.DataFrame, start_idx: int, end_idx: int,
                       content_name: str, market_reference: str, model: str, temperature: float,
                       max_output_tokens: int, top_p: float, top_k: int) -> List[Dict[str, Any]]:
    """
    Process a batch of rows concurrently.
    """
    tasks = []
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        if not row[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']].isnull().any():
            continue
        task = process_row(row, idx, content_name, market_reference, model, temperature,
                         max_output_tokens, top_p, top_k)
        tasks.append(task)
    return await asyncio.gather(*tasks)

async def llm_features(df: pd.DataFrame, column_name: str, market_reference: str = "S&P 500",
                      model: str = "gemini-2.0-flash-lite", temperature: float = 0.0,
                      max_output_tokens: int = 100, top_p: float = 0.1, top_k: int = 1) -> pd.DataFrame:
    """
    Process a DataFrame to add LLM-generated features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column containing text to analyze
        market_reference (str): Reference market for relevance scoring
        model (str): Model to use for generation
        temperature (float): Temperature for generation
        max_output_tokens (int): Maximum output tokens
        top_p (float): Top p for generation
        top_k (int): Top k for generation
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    # Initialize new columns if they don't exist
    for col in ['sentiment_score', 'relevance_score', 'event_importance', 'event_type']:
        if col not in df.columns:
            df[col] = None

    total_rows = len(df)
    logger.info(f"Total rows to process: {total_rows}")
    logger.info(f"Using {MAX_CONCURRENT_TASKS} concurrent tasks (API AFC limit)")

    for batch_start in range(0, total_rows, MAX_CONCURRENT_TASKS):
        batch_end = min(batch_start + MAX_CONCURRENT_TASKS, total_rows)
        logger.info(f"Processing batch {batch_start} to {batch_end}")
        results = await process_batch(df, batch_start, batch_end, column_name, market_reference,
                                   model, temperature, max_output_tokens, top_p, top_k)
        
        for result in results:
            idx = result['idx']
            if 'error' not in result:
                df.at[idx, 'sentiment_score'] = result.get('sentiment_score')
                df.at[idx, 'relevance_score'] = result.get('relevance_score')
                df.at[idx, 'event_importance'] = result.get('event_importance')
                df.at[idx, 'event_type'] = result.get('event_type')

    return df

async def test_llm_features():
    """
    Test function to demonstrate llm_features functionality.
    Reads first 100 rows from Bloomberg data and processes them.
    """
    # Read the first 100 rows
    input_path = "data/processed/bloomberg/sentiment_inserted.csv"
    logger.info(f"Reading first 100 rows from {input_path}")
    df = pd.read_csv(input_path, nrows=100)
    
    # Print initial state
    logger.info("Initial DataFrame head:")
    print("\nInitial DataFrame head:")
    print(df.head())
    
    # Process the data
    logger.info("Processing data with llm_features...")
    df = await llm_features(df, column_name='headline', market_reference='S&P 500')
    
    # Print results
    logger.info("Processed DataFrame head:")
    print("\nProcessed DataFrame head:")
    print(df.head())
    
    # Print summary statistics
    logger.info("Summary statistics of generated features:")
    print("\nSummary statistics:")
    print(df[['sentiment_score', 'relevance_score', 'event_importance']].describe())
    
    # Print event type distribution
    logger.info("Event type distribution:")
    print("\nEvent type distribution:")
    print(df['event_type'].value_counts())

if __name__ == '__main__':
    asyncio.run(test_llm_features()) 