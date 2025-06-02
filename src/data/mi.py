import json
import aiohttp
import asyncio
import os
import logging
import pandas as pd
import ssl
import certifi
from datetime import datetime, timedelta
from typing import List, Optional
from src.utils.logger import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)

DATA_PATH = "data/test.csv"

# Load environment variables from .env file
load_dotenv()

class MIData():
    def __init__(self):
        self.mi_df = None
        self._import_Data()
        # self._process()
    
    def _import_Data(self):
        """Import market impact data from local file or fetch from API if not available."""
        try:
            # Check if local file exists
            if os.path.exists(DATA_PATH):
                logger.info(f"Loading market impact data from {DATA_PATH}")
                self.mi_df = pd.read_csv(DATA_PATH)
                logger.info(f"Successfully loaded data with shape: {self.mi_df.shape}")
            else:
                logger.info("Local market impact data not found. Fetching from API...")
                # Create data directory if it doesn't exist
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                
                # Fetch data from API
                self._fetch()
                
                # # Save the fetched data
                # if self.mi_df is not None and not self.mi_df.empty:
                #     self.mi_df.to_csv(DATA_PATH, index=False)
                #     logger.info(f"Saved market impact data to {DATA_PATH}")
                # else:
                #     logger.error("Failed to fetch market impact data from API")
        except Exception as e:
            logger.error(f"Error in _import_Data: {str(e)}")
            raise
    
    def _process(self):
        # Sort by date
        self.mi_df = self.mi_df.sort_values('date')
        logger.info("Sorted by date")

        # Convert datetime to UTC
        self.mi_df['date'] = pd.to_datetime(self.mi_df['date'], utc=True)
    
    def _fetch(self):
        series_list = [
            'MRTSSM44X72USS',  # Consumer: Retail Sales
            'PAYEMS',          # Employment: Total Nonfarm Payroll
            'HOUSTPFST1FQ',    # Housing: Housing Starts One Family
            'CPALTT01USM657N', # Prices: CPI
            'CORESTICKM159SFRBATL', # Prices: Core CPI
            'GDP',             # Producer: GDP
            'INDPRO',          # Producer: Industrial Production
            'BSCURT02USM160S'  # Producer: Capacity Utilisation
       ]
        self.mi_df = asyncio.run(_download_multiple_series(series_list))
        
    def get_mi_df(self) -> pd.DataFrame:
        return self.mi_df.copy()
    
class StouisfedFetcher:
    def __init__(self):
        self.api_key = os.getenv('FRED_API_KEY')
        # Create SSL context with proper certificates
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def fetch_data(self, series_id: str, period: str):
        url = 'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        if period == '1y':
            one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            params['observation_start'] = one_year_ago
        
        # Create connector with SSL context
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:  
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logging.error(f'Error({response.status}) while fetching stlousfed data: {series_id}')
                    return None

    async def get_df(self, series_id: str, period: str = None) -> pd.DataFrame:
        json_data = await self.fetch_data(series_id, period)
        if json_data is not None and 'observations' in json_data:
            observations = json_data['observations']
            dates = [obs['date'] for obs in observations]
            values = [float(obs['value']) if obs['value'] != '.' else None for obs in observations]
            df = pd.DataFrame({'date': dates, series_id: values})
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return None

async def _download_multiple_series(series_list: List[str], period: Optional[str] = None) -> pd.DataFrame:
    """
    Download multiple series from FRED and combine them into a single DataFrame.
    
    Args:
        api_key: FRED API key
        series_list: List of series IDs to download
        period: Optional period filter (e.g., '1y' for last year)
    
    Returns:
        DataFrame with dates as index and series IDs as columns
    """
    fetcher = StouisfedFetcher()
    
    # Download all series concurrently
    tasks = [fetcher.get_df(series_id, period) for series_id in series_list]
    dfs = await asyncio.gather(*tasks)
    
    # Filter out None results and merge all DataFrames
    valid_dfs = [df for df in dfs if df is not None]
    if not valid_dfs:
        return pd.DataFrame()
    
    # Merge all DataFrames on date
    final_df = valid_dfs[0]
    for df in valid_dfs[1:]:
        final_df = pd.merge(final_df, df, on='date', how='outer')
    
    # Set date as index and sort
    final_df.set_index('date', inplace=True)
    final_df.sort_index(inplace=True)
    
    # Create a complete daily date range
    date_range = pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq='D')
    final_df = final_df.reindex(date_range)
    
    # Apply linear interpolation for each column
    for column in final_df.columns:
        # Get the original frequency of the data
        original_data = final_df[column].dropna()
        if len(original_data) > 0:
            # Calculate the average number of days between observations
            avg_days = (original_data.index[-1] - original_data.index[0]).days / (len(original_data) - 1)
            
            # If the data is quarterly (avg_days > 60) or monthly (avg_days > 25), use linear interpolation
            if avg_days > 25:
                final_df[column] = final_df[column].interpolate(method='linear')
    
    # Trim data to specific date range after interpolation
    start_date = pd.to_datetime('2006-10-20')
    end_date = pd.to_datetime('2013-11-26')
    final_df = final_df.loc[start_date:end_date]
    
    return final_df
