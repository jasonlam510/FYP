import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATA_PATH = "data/finbert_llm_sentiment_inserted.csv"

class NewsData():
    def __init__(self):
        self.news_df = None
        self._import_Data()
        self._process()
        self._split_data()

    
    def _import_Data(self):
        logger.info(f"Import news data from: {DATA_PATH}")
        self.news_df = pd.read_csv(DATA_PATH)
        logger.info(f"Successfuly imported: {self.news_df.shape}")
    
    def _process(self):
        # Sort by date
        self.news_df = self.news_df.sort_values('date')
        logger.info("sorted by date")

        # Convert to UTC
        self.news_df['date'] = pd.to_datetime(self.news_df['date'], utc=True)
        logger.info("datetime is converted to UTC")
    
    def get_finbert_df(self) -> pd.DataFrame:
        return self.finbert_df['date', 'sentiment_score_finbert', 'sentiment_positive_finbert', 'sentiment_neutral_finbert', 'sentiment_negative_finbert'].copy()
    
    def get_llm_df(self) -> pd.DataFrame:
        return self.llm_df['date', 'sentiment_score_llm', 'relevance_score','event_importance','event_type'].copy()
