import pandas as pd
from src.utils.logger import get_logger

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import CCIIndicator
from typing import List, Dict

logger = get_logger(__name__)

DATA_PATH = "data/SPX_1d.csv"

class PriceData():
    def __init__(self):
        self.price_df = None
        self._import_Data()
        self._process()
    
    def _import_Data(self):
        logger.info(f"Import news data from: {DATA_PATH}")
        self.price_df = pd.read_csv(DATA_PATH)
        logger.info(f"Successfuly imported: {self.price_df.shape}")
    
    def _process(self):

        # Sort by date
        self.price_df = self.price_df.sort_values('date')
        logger.info("Sorted by date")

        # Convert datetime to UTC
        self.price_df['date'] = pd.to_datetime(self.price_df['date'], utc=True)
    
    def get_price_df(self) -> pd.DataFrame:
        return self.price_df.copy()
    
