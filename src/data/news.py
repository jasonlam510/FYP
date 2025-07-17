import pandas as pd
from src.utils.logger import get_logger
from src.config import NEWS_DATA_PATH

logger = get_logger(__name__)

class NewsData():
    def __init__(self):
        self.news_df = None
        self.finbert_df = None
        self.llm_df = None
        self._import_Data()
        self._process()
        self._split_data()
    
    def _import_Data(self):
        logger.info(f"Import news data from: {NEWS_DATA_PATH}")
        self.news_df = pd.read_csv(NEWS_DATA_PATH)
        logger.info(f"Successfully imported: {self.news_df.shape}")
    
    def _process(self):
        # Sort by date
        self.news_df = self.news_df.sort_values('date')
        logger.info("sorted by date")

        # Convert to UTC and normalize to midnight
        self.news_df['date'] = pd.to_datetime(self.news_df['date'], utc=True).dt.normalize()
        logger.info("datetime is converted to UTC and normalized to midnight")
    
    def _split_data(self):
        """Split the data into FinBERT and LLM datasets."""
        # Create FinBERT dataset
        finbert_columns = ['date', 'sentiment_score_finbert', 'sentiment_positive_finbert', 
                         'sentiment_neutral_finbert', 'sentiment_negative_finbert']
        self.finbert_df = self.news_df[finbert_columns].copy()
        
        # Create LLM dataset
        llm_columns = ['date', 'sentiment_score_llm', 'relevance_score', 
                      'event_importance', 'event_type']
        self.llm_df = self.news_df[llm_columns].copy()
        
        logger.info(f"Split data into FinBERT ({self.finbert_df.shape}) and LLM ({self.llm_df.shape}) datasets")
    
    def get_finbert_df(self) -> pd.DataFrame:
        """Get the FinBERT dataset."""
        return self.finbert_df.copy()
    
    def get_llm_df(self) -> pd.DataFrame:
        """Get LLM sentiment data with all features."""
        return self.llm_df.copy()

    def get_llm_sentiment_df(self) -> pd.DataFrame:
        """Get LLM sentiment data with only sentiment scores."""
        # Select only date and sentiment columns
        return self.llm_df[['date', 'sentiment_score_llm']].copy()
