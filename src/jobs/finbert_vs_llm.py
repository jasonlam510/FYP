import pandas as pd
import numpy as np

import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "finbert_vs_llm_comparison"
os.environ['JOB_NAME'] = JOB_NAME

from src.data.news import NewsData
from src.data.price import PriceData
from src.model.train_model import train_model, plot_results, build_cnn_lstm_model, build_lstm_model
from src.utils.combine import (
    aggregate_news_price_rolling_finbert,
    aggregate_news_price_rolling_llm
)
from src.utils.logger import get_logger
from src.config import (
    EVENT_TYPES,
    SEQ_LENGTH,
    N_TRIALS,
    HALF_LIFE_RANGE,
    N_DAYS_RANGE
)

logger = get_logger(__name__)

def run_comparison():
    # Load data
    logger.info("Loading data...")
    news_data = NewsData()
    price_data = PriceData()

    # Get FinBERT and LLM datasets
    finbert_df = news_data.get_finbert_df()
    llm_df = news_data.get_llm_df()
    price_df = price_data.get_price_df()

    # Dictionary to store results
    results = {}

    # 1. Train LSTM with FinBERT
    logger.info("Training LSTM with FinBERT data...")
    finbert_aggregated = aggregate_news_price_rolling_finbert(
        finbert_df, price_df, n_days=3, half_life_days=1.5
    )
    lstm_finbert_model, lstm_finbert_history, lstm_finbert_pred, lstm_finbert_actual, test_dates = train_model(
        finbert_df, price_df, SEQ_LENGTH, N_TRIALS, build_model_fn=build_lstm_model
    )
    results['LSTM-FinBERT'] = {
        'predictions': lstm_finbert_pred,
        'actual': lstm_finbert_actual,
        'test_dates': test_dates
    }

    # 2. Train LSTM-CNN with FinBERT
    logger.info("Training LSTM-CNN with FinBERT data...")
    lstm_cnn_finbert_model, lstm_cnn_finbert_history, lstm_cnn_finbert_pred, lstm_cnn_finbert_actual, test_dates = train_model(
        finbert_df, price_df, SEQ_LENGTH, N_TRIALS, build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-FinBERT'] = {
        'predictions': lstm_cnn_finbert_pred,
        'actual': lstm_cnn_finbert_actual,
        'test_dates': test_dates
    }

    # 3. Train LSTM with LLM
    logger.info("Training LSTM with LLM data...")
    llm_aggregated = aggregate_news_price_rolling_llm(
        llm_df, price_df, EVENT_TYPES, n_days=3, half_life_days=1.5
    )
    lstm_llm_model, lstm_llm_history, lstm_llm_pred, lstm_llm_actual, test_dates = train_model(
        llm_df, price_df, SEQ_LENGTH, N_TRIALS, build_model_fn=build_lstm_model
    )
    results['LSTM-LLM'] = {
        'predictions': lstm_llm_pred,
        'actual': lstm_llm_actual,
        'test_dates': test_dates
    }

    # 4. Train LSTM-CNN with LLM
    logger.info("Training LSTM-CNN with LLM data...")
    lstm_cnn_llm_model, lstm_cnn_llm_history, lstm_cnn_llm_pred, lstm_cnn_llm_actual, test_dates = train_model(
        llm_df, price_df, SEQ_LENGTH, N_TRIALS, build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-LLM'] = {
        'predictions': lstm_cnn_llm_pred,
        'actual': lstm_cnn_llm_actual,
        'test_dates': test_dates
    }

    # Plot results
    predictions_list = [
        results['LSTM-FinBERT']['predictions'],
        results['LSTM-CNN-FinBERT']['predictions'],
        results['LSTM-LLM']['predictions'],
        results['LSTM-CNN-LLM']['predictions']
    ]
    line_names = ['LSTM-FinBERT', 'LSTM-CNN-FinBERT', 'LSTM-LLM', 'LSTM-CNN-LLM']
    actual = results['LSTM-FinBERT']['actual']  # All actual values should be the same
    test_dates = results['LSTM-FinBERT']['test_dates']

    plot_results(
        predictions_list=predictions_list,
        line_names=line_names,
        actual=actual,
        test_dates=test_dates,
        job_name=JOB_NAME
    )

    # Calculate and log metrics
    for model_name, result in results.items():
        mse = np.mean((result['predictions'] - result['actual']) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(result['predictions'] - result['actual']))
        logger.info(f"\nMetrics for {model_name}:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    run_comparison() 