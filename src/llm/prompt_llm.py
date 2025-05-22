import os
import pandas as pd
import json
import signal
from dotenv import load_dotenv
import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Any
import time
sys.path.append(str(Path.cwd()))
from src.llm.google_genai_client import generate_response
from tqdm import tqdm
import logging

# Configure logging
log_file_path = os.path.join('src', 'llm', 'logs', 'bloomberg.log')
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_file_path,
                    filemode='a')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MARKET_REFERENCE = os.getenv('MARKET_REFERENCE', 'S&P 500')
SAVE_INTERVAL = 10000
MAX_CONCURRENT_TASKS = 500 # Match API's AFC limit
MAX_RPM = 4000  # Maximum requests per minute
REQUEST_INTERVAL = 60 / MAX_RPM  # Time to wait between requests in seconds

class RateLimiter:
    def __init__(self, max_rpm: int):
        self.max_rpm = max_rpm
        self.interval = 60 / max_rpm
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.request_count = 0
        self.start_time = time.time()
        self.active_requests = 0
        self.max_active_requests = MAX_CONCURRENT_TASKS
        self.request_lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.interval:
                await asyncio.sleep(self.interval - time_since_last_request)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            # Log performance metrics every 100 requests
            if self.request_count % 100 == 0:
                elapsed_time = current_time - self.start_time
                requests_per_second = self.request_count / elapsed_time
                logger.info(f"Performance metrics - Requests: {self.request_count}, "
                          f"Elapsed time: {elapsed_time:.2f}s, "
                          f"Requests/second: {requests_per_second:.2f}, "
                          f"Active requests: {self.active_requests}")

    async def start_request(self):
        async with self.request_lock:
            while self.active_requests >= self.max_active_requests:
                await asyncio.sleep(0.1)  # Wait if we're at the limit
            self.active_requests += 1

    async def end_request(self):
        async with self.request_lock:
            self.active_requests -= 1

# Initialize rate limiter
rate_limiter = RateLimiter(MAX_RPM)

# Save the CSV when interrupted
def save_on_interrupt(signal, frame, df, output_csv):
    print("Saving progress before exit...")
    df.to_csv(output_csv, index=False)
    print(f"Annotated CSV saved to {output_csv}")
    exit(0)

def build_prompt(content: str, content_name: str = "news") -> str:
    """
    Construct a generic prompt for Gemini based on arbitrary news content.
    """
    return (
        f"You are a financial‐news analysis assistant. Given only the {content_name} of the news, you must output a valid JSON object with exactly these four fields (no extra keys, no prose):\n"
        f"- sentiment_score: float between -1 (very negative) and 1 (very positive)  \n"
        f"- relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the {MARKET_REFERENCE})  \n"
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

async def process_row(row: pd.Series, idx: int, content_name: str,
                     model: str, temperature: float, max_output_tokens: int, top_p: float, top_k: int) -> Dict[str, Any]:
    """
    Process a single row asynchronously.
    """
    content = row.get(content_name, '') or row.get(content_name.capitalize(), '')
    if not isinstance(content, str) or not content.strip():
        return {'idx': idx, 'error': 'Empty content'}

    prompt = build_prompt(content, content_name)
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
                       content_name: str, model: str, temperature: float, max_output_tokens: int,
                       top_p: float, top_k: int) -> List[Dict[str, Any]]:
    """
    Process a batch of rows concurrently.
    """
    tasks = []
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        if not row[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']].isnull().any():
            continue
        task = process_row(row, idx, content_name, model, temperature, max_output_tokens, top_p, top_k)
        tasks.append(task)
    return await asyncio.gather(*tasks)

async def annotate_content_async(input_csv: str, output_csv: str,
                               model: str = "gemini-2.0-flash-lite",
                               temperature: float = 0.0,
                               max_output_tokens: int = 100,
                               top_p: float = 0.1,
                               top_k: int = 1,
                               content_name: str = "headline"):
    """
    Asynchronous version of annotate_content.
    """
    # Set up the signal handler
    signal.signal(signal.SIGINT, lambda signal, frame: save_on_interrupt(signal, frame, df, output_csv))

    # Load or create DataFrame
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        unprocessed_rows = df[df[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']].isnull().any(axis=1)]
        start_idx = unprocessed_rows.index.min()
    else:
        df = pd.read_csv(input_csv)
        df[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']] = None
        df.to_csv(output_csv, index=False)
        start_idx = 0

    total_rows = len(df)
    logger.info(f"Total rows to process: {total_rows}")
    logger.info(f"Start index: {start_idx}")
    logger.info(f"Using {MAX_CONCURRENT_TASKS} concurrent tasks (API AFC limit)")
    counter = 0

    for batch_start in tqdm(range(start_idx, total_rows, MAX_CONCURRENT_TASKS), desc="Processing batches"):
        batch_end = min(batch_start + MAX_CONCURRENT_TASKS, total_rows)
        results = await process_batch(df, batch_start, batch_end, content_name,
                                   model, temperature, max_output_tokens, top_p, top_k)
        
        for result in results:
            idx = result['idx']
            if 'error' not in result:
                df.at[idx, 'sentiment_score'] = result.get('sentiment_score')
                df.at[idx, 'relevance_score'] = result.get('relevance_score')
                df.at[idx, 'event_importance'] = result.get('event_importance')
                df.at[idx, 'event_type'] = result.get('event_type')
                counter += 1
                if counter % SAVE_INTERVAL == 0:
                    df.to_csv(output_csv, index=False)
                    logger.info(f"Annotated CSV saved to {output_csv}")

    df.to_csv(output_csv, index=False)
    logger.info(f"Annotated CSV saved to {output_csv}")

def annotate_content(input_csv: str, output_csv: str,
                    model: str = "gemini-2.0-flash-lite",
                    temperature: float = 0.0,
                    max_output_tokens: int = 100,
                    top_p: float = 0.1,
                    top_k: int = 1,
                    content_name: str = "headline"):
    """
    Wrapper function to run the async version.
    """
    asyncio.run(annotate_content_async(input_csv, output_csv, model, temperature,
                                     max_output_tokens, top_p, top_k, content_name))

def run_on_bloomberg():
    input_csv = "data/processed/bloomberg/sentiment_inserted.csv"
    output_csv = "data/processed/bloomberg/llm_bloomberg.csv"
    annotate_content(input_csv, output_csv, content_name="Headline")

if __name__ == '__main__':
    run_on_bloomberg()
