import pandas as pd
from src.utils.logger import get_logger
from src.config import PRICE_DATA_PATH

logger = get_logger(__name__)

class PriceData():
    def __init__(self):
        self.price_df = None
        self._import_Data()
        self._process()
    
    def _import_Data(self):
        logger.info(f"Import price data from: {PRICE_DATA_PATH}")
        self.price_df = pd.read_csv(PRICE_DATA_PATH)
        logger.info(f"Successfully imported: {self.price_df.shape}")
    
    def _process(self):
        # Lowercase all column names
        self.price_df.columns = self.price_df.columns.str.lower()
        logger.info("Column names are converted to lowercase")
        
        # Sort by date
        self.price_df = self.price_df.sort_values('date')
        logger.info("Sorted by date")

        # Convert datetime to UTC and normalize to midnight
        self.price_df['date'] = pd.to_datetime(self.price_df['date'], utc=True).dt.normalize()
        logger.info("datetime is converted to UTC and normalized to midnight")
    
    def get_price_df(self) -> pd.DataFrame:
        """Get the price dataset."""
        return self.price_df.copy()
    
