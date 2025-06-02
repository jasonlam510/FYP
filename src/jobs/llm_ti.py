import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "llm_ti"
os.environ['JOB_NAME'] = JOB_NAME

from src.data.news import NewsData
from src.data.price import PriceData
from src.model.train_model import train_model, plot_results, build_cnn_lstm_model, build_lstm_model
from src.utils.combine import aggregate_news_price_rolling_llm
from src.helper.ti import calculate_technical_indicators
from src.utils.logger import get_logger
from src.config import (
    EVENT_TYPES,
    SEQ_LENGTH,
    N_TRIALS,
    BALANCED_INDICATORS,
    MULTI_TIMEFRAME_INDICATORS
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

    # Get LLM dataset
    llm_df = news_data.get_llm_df()
    price_df = price_data.get_price_df()

    # Dictionary to store results
    results = {}

    # 1. Train LSTM with LLM + Balanced TI
    logger.info("Training LSTM with LLM + Balanced TI...")
    price_df_balanced = calculate_technical_indicators(price_df.copy(), BALANCED_INDICATORS)
    lstm_balanced_model, lstm_balanced_history, lstm_balanced_pred, lstm_balanced_actual, test_dates, lstm_balanced_metrics, lstm_balanced_params = train_model(
        llm_df, price_df_balanced, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model
    )
    results['LSTM-LLM-Balanced-TI'] = {
        'predictions': lstm_balanced_pred,
        'actual': lstm_balanced_actual,
        'test_dates': test_dates,
        'metrics': lstm_balanced_metrics,
        'best_params': lstm_balanced_params
    }

    # 2. Train LSTM with LLM + Multi-timeframe TI
    logger.info("Training LSTM with LLM + Multi-timeframe TI...")
    price_df_multi = calculate_technical_indicators(price_df.copy(), MULTI_TIMEFRAME_INDICATORS)
    lstm_multi_model, lstm_multi_history, lstm_multi_pred, lstm_multi_actual, test_dates, lstm_multi_metrics, lstm_multi_params = train_model(
        llm_df, price_df_multi, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model
    )
    results['LSTM-LLM-Multi-TI'] = {
        'predictions': lstm_multi_pred,
        'actual': lstm_multi_actual,
        'test_dates': test_dates,
        'metrics': lstm_multi_metrics,
        'best_params': lstm_multi_params
    }

    # 3. Train LSTM-CNN with LLM + Balanced TI
    logger.info("Training LSTM-CNN with LLM + Balanced TI...")
    lstm_cnn_balanced_model, lstm_cnn_balanced_history, lstm_cnn_balanced_pred, lstm_cnn_balanced_actual, test_dates, lstm_cnn_balanced_metrics, lstm_cnn_balanced_params = train_model(
        llm_df, price_df_balanced, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-LLM-Balanced-TI'] = {
        'predictions': lstm_cnn_balanced_pred,
        'actual': lstm_cnn_balanced_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_balanced_metrics,
        'best_params': lstm_cnn_balanced_params
    }

    # 4. Train LSTM-CNN with LLM + Multi-timeframe TI
    logger.info("Training LSTM-CNN with LLM + Multi-timeframe TI...")
    lstm_cnn_multi_model, lstm_cnn_multi_history, lstm_cnn_multi_pred, lstm_cnn_multi_actual, test_dates, lstm_cnn_multi_metrics, lstm_cnn_multi_params = train_model(
        llm_df, price_df_multi, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-LLM-Multi-TI'] = {
        'predictions': lstm_cnn_multi_pred,
        'actual': lstm_cnn_multi_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_multi_metrics,
        'best_params': lstm_cnn_multi_params
    }

    # Plot results
    predictions_list = [
        results['LSTM-LLM-Balanced-TI']['predictions'],
        results['LSTM-LLM-Multi-TI']['predictions'],
        results['LSTM-CNN-LLM-Balanced-TI']['predictions'],
        results['LSTM-CNN-LLM-Multi-TI']['predictions']
    ]
    line_names = [
        'LSTM-LLM-Balanced-TI',
        'LSTM-LLM-Multi-TI',
        'LSTM-CNN-LLM-Balanced-TI',
        'LSTM-CNN-LLM-Multi-TI'
    ]
    actual = results['LSTM-LLM-Balanced-TI']['actual']  # All actual values should be the same
    test_dates = results['LSTM-LLM-Balanced-TI']['test_dates']
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