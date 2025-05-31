import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "llm_technical_indicators_comparison"
os.environ['JOB_NAME'] = JOB_NAME

from src.data.news import NewsData
from src.data.price import PriceData
from src.model.train_model import train_model, plot_results, build_cnn_lstm_model, build_lstm_model
from src.utils.combine import aggregate_news_price_rolling_llm
from src.utils.logger import get_logger
from src.config import (
    EVENT_TYPES,
    SEQ_LENGTH,
    N_TRIALS,
    HALF_LIFE_RANGE,
    N_DAYS_RANGE,
    BALANCED_INDICATORS,
    MULTI_TIMEFRAME_INDICATORS
)
from src.helper.ti import calculate_technical_indicators

logger = get_logger(__name__)

def find_best_parameters(llm_df, price_df, event_types):
    """Find the best n_days and half_life_days parameters using grid search."""
    best_mse = float('inf')
    best_params = None
    
    for n_days in N_DAYS_RANGE:
        for half_life in HALF_LIFE_RANGE:
            try:
                aggregated = aggregate_news_price_rolling_llm(
                    llm_df, price_df, event_types, n_days=n_days, half_life_days=half_life
                )
                # Use a simple model to evaluate the parameters
                model, history, pred, actual, _ = train_model(
                    llm_df, price_df, SEQ_LENGTH, 1, build_model_fn=build_lstm_model
                )
                mse = np.mean((pred - actual) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_params = (n_days, half_life)
                    
                logger.info(f"Parameters: n_days={n_days}, half_life={half_life}, MSE={mse:.4f}")
            except Exception as e:
                logger.warning(f"Failed for parameters n_days={n_days}, half_life={half_life}: {str(e)}")
                continue
    
    logger.info(f"Best parameters found: n_days={best_params[0]}, half_life={best_params[1]}")
    return best_params

def run_comparison():
    # Load data
    logger.info("Loading data...")
    news_data = NewsData()
    price_data = PriceData()

    # Get LLM dataset and price data
    llm_df = news_data.get_llm_df()
    price_df = price_data.get_price_df()

    # Calculate technical indicators for both sets
    logger.info("Calculating balanced technical indicators...")
    price_df_balanced = calculate_technical_indicators(price_df.copy(), BALANCED_INDICATORS)
    
    logger.info("Calculating multi-timeframe technical indicators...")
    price_df_multi = calculate_technical_indicators(price_df.copy(), MULTI_TIMEFRAME_INDICATORS)

    # Find best parameters for each indicator set
    logger.info("Finding best parameters for balanced indicators...")
    best_params_balanced = find_best_parameters(llm_df, price_df_balanced, EVENT_TYPES)
    
    logger.info("Finding best parameters for multi-timeframe indicators...")
    best_params_multi = find_best_parameters(llm_df, price_df_multi, EVENT_TYPES)

    # Dictionary to store results
    results = {}

    # 1. Train LSTM with Balanced Indicators
    logger.info("Training LSTM with Balanced Indicators...")
    balanced_aggregated = aggregate_news_price_rolling_llm(
        llm_df, price_df_balanced, EVENT_TYPES, 
        n_days=best_params_balanced[0], 
        half_life_days=best_params_balanced[1]
    )
    lstm_balanced_model, lstm_balanced_history, lstm_balanced_pred, lstm_balanced_actual, test_dates = train_model(
        llm_df, price_df_balanced, SEQ_LENGTH, N_TRIALS, build_model_fn=build_lstm_model
    )
    results['LSTM-Balanced'] = {
        'predictions': lstm_balanced_pred,
        'actual': lstm_balanced_actual,
        'test_dates': test_dates,
        'params': best_params_balanced
    }

    # 2. Train LSTM-CNN with Balanced Indicators
    logger.info("Training LSTM-CNN with Balanced Indicators...")
    lstm_cnn_balanced_model, lstm_cnn_balanced_history, lstm_cnn_balanced_pred, lstm_cnn_balanced_actual, test_dates = train_model(
        llm_df, price_df_balanced, SEQ_LENGTH, N_TRIALS, build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-Balanced'] = {
        'predictions': lstm_cnn_balanced_pred,
        'actual': lstm_cnn_balanced_actual,
        'test_dates': test_dates,
        'params': best_params_balanced
    }

    # 3. Train LSTM with Multi-timeframe Indicators
    logger.info("Training LSTM with Multi-timeframe Indicators...")
    multi_aggregated = aggregate_news_price_rolling_llm(
        llm_df, price_df_multi, EVENT_TYPES, 
        n_days=best_params_multi[0], 
        half_life_days=best_params_multi[1]
    )
    lstm_multi_model, lstm_multi_history, lstm_multi_pred, lstm_multi_actual, test_dates = train_model(
        llm_df, price_df_multi, SEQ_LENGTH, N_TRIALS, build_model_fn=build_lstm_model
    )
    results['LSTM-Multi'] = {
        'predictions': lstm_multi_pred,
        'actual': lstm_multi_actual,
        'test_dates': test_dates,
        'params': best_params_multi
    }

    # 4. Train LSTM-CNN with Multi-timeframe Indicators
    logger.info("Training LSTM-CNN with Multi-timeframe Indicators...")
    lstm_cnn_multi_model, lstm_cnn_multi_history, lstm_cnn_multi_pred, lstm_cnn_multi_actual, test_dates = train_model(
        llm_df, price_df_multi, SEQ_LENGTH, N_TRIALS, build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-Multi'] = {
        'predictions': lstm_cnn_multi_pred,
        'actual': lstm_cnn_multi_actual,
        'test_dates': test_dates,
        'params': best_params_multi
    }

    # Plot results
    predictions_list = [
        results['LSTM-Balanced']['predictions'],
        results['LSTM-CNN-Balanced']['predictions'],
        results['LSTM-Multi']['predictions'],
        results['LSTM-CNN-Multi']['predictions']
    ]
    line_names = ['LSTM-Balanced', 'LSTM-CNN-Balanced', 'LSTM-Multi', 'LSTM-CNN-Multi']
    actual = results['LSTM-Balanced']['actual']  # All actual values should be the same
    test_dates = results['LSTM-Balanced']['test_dates']

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
        logger.info(f"Parameters: n_days={result['params'][0]}, half_life={result['params'][1]}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    run_comparison() 