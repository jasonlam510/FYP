import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "sentiment_comparison"
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

def print_best_hyperparameters(results):
    """Print the best hyperparameters for each model."""
    logger.info("\n=== Best Hyperparameters for Each Model ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        for param_name, param_value in result['best_params'].items():
            logger.info(f"{param_name}: {param_value}")

def run_comparison():
    # Load data
    logger.info("Loading data...")
    news_data = NewsData()
    price_data = PriceData()

    # Get FinBERT and LLM sentiment datasets
    finbert_df = news_data.get_finbert_df()
    llm_sentiment_df = news_data.get_llm_sentiment_df()
    price_df = price_data.get_price_df()

    # Dictionary to store results
    results = {}


    # 4. Train LSTM-CNN with LLM Sentiment
    logger.info("Training LSTM-CNN with LLM sentiment data...")
    lstm_cnn_llm_model, lstm_cnn_llm_history, lstm_cnn_llm_pred, lstm_cnn_llm_actual, test_dates, lstm_cnn_llm_metrics, lstm_cnn_llm_params = train_model(
        llm_sentiment_df, price_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model,
        half_life_range=HALF_LIFE_RANGE,
        n_days_range=N_DAYS_RANGE
    )
    results['LSTM-CNN-LLM-Sentiment'] = {
        'predictions': lstm_cnn_llm_pred,
        'actual': lstm_cnn_llm_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_llm_metrics,
        'best_params': lstm_cnn_llm_params,
    }

    # 1. Train LSTM with FinBERT
    logger.info("Training LSTM with FinBERT data...")
    lstm_finbert_model, lstm_finbert_history, lstm_finbert_pred, lstm_finbert_actual, test_dates, lstm_finbert_metrics, lstm_finbert_params = train_model(
        finbert_df, price_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model,
        half_life_range=HALF_LIFE_RANGE,
        n_days_range=N_DAYS_RANGE
    )
    results['LSTM-FinBERT'] = {
        'predictions': lstm_finbert_pred,
        'actual': lstm_finbert_actual,
        'test_dates': test_dates,
        'metrics': lstm_finbert_metrics,
        'best_params': lstm_finbert_params,
    }

    # 2. Train LSTM-CNN with FinBERT
    logger.info("Training LSTM-CNN with FinBERT data...")
    lstm_cnn_finbert_model, lstm_cnn_finbert_history, lstm_cnn_finbert_pred, lstm_cnn_finbert_actual, test_dates, lstm_cnn_finbert_metrics, lstm_cnn_finbert_params = train_model(
        finbert_df, price_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model,
        half_life_range=HALF_LIFE_RANGE,
        n_days_range=N_DAYS_RANGE
    )
    results['LSTM-CNN-FinBERT'] = {
        'predictions': lstm_cnn_finbert_pred,
        'actual': lstm_cnn_finbert_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_finbert_metrics,
        'best_params': lstm_cnn_finbert_params
    }

    # 3. Train LSTM with LLM Sentiment
    logger.info("Training LSTM with LLM sentiment data...")
    lstm_llm_model, lstm_llm_history, lstm_llm_pred, lstm_llm_actual, test_dates, lstm_llm_metrics, lstm_llm_params = train_model(
        llm_sentiment_df, price_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model,
        half_life_range=HALF_LIFE_RANGE,
        n_days_range=N_DAYS_RANGE
    )
    results['LSTM-LLM-Sentiment'] = {
        'predictions': lstm_llm_pred,
        'actual': lstm_llm_actual,
        'test_dates': test_dates,
        'metrics': lstm_llm_metrics,
        'best_params': lstm_llm_params
    }

    # Plot results
    predictions_list = [
        results['LSTM-FinBERT']['predictions'],
        results['LSTM-CNN-FinBERT']['predictions'],
        results['LSTM-LLM-Sentiment']['predictions'],
        results['LSTM-CNN-LLM-Sentiment']['predictions']
    ]
    line_names = ['LSTM-FinBERT', 'LSTM-CNN-FinBERT', 'LSTM-LLM-Sentiment', 'LSTM-CNN-LLM-Sentiment']
    actual = results['LSTM-FinBERT']['actual']  # All actual values should be the same
    test_dates = results['LSTM-FinBERT']['test_dates']
    metrics_list = [results[model]['metrics'] for model in line_names]

    plot_results(
        predictions_list=predictions_list,
        line_names=line_names,
        actual=actual,
        test_dates=test_dates,
        metrics_list=metrics_list,
        job_name=JOB_NAME
    )

    # Print best hyperparameters
    print_best_hyperparameters(results)

    # Log detailed metrics for each model
    for model_name, result in results.items():
        metrics = result['metrics']
        logger.info(f"\nDetailed Metrics for {model_name}:")
        logger.info(f"RMSE = ${metrics['RMSE']:.2f}, MAE = ${metrics['MAE']:.2f} on next-day close")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        logger.info(f"DC Precision: {metrics['dc_precision']:.2%}")
        logger.info(f"DC Recall: {metrics['dc_recall']:.2%}")
        logger.info(f"DC Timing Error: {metrics['dc_timing_error']:.2f} days")

if __name__ == "__main__":
    run_comparison() 