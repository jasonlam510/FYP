import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Callable, Union
from src.utils.logger import get_logger
from src.utils.combine import aggregate_news_price_rolling_finbert, aggregate_news_price_rolling_llm, aggregate_news_price_rolling_llm_sentiment
from src.config import EVENT_TYPES
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.optuna_storage import get_study
import csv

logger = get_logger(__name__)

def create_sequences(data, seq_length, target_col_idx):
    """Create sequences for time series data."""
    logger.info(f"[DEBUG] create_sequences: data.shape = {data.shape}")
    logger.info(f"[DEBUG] create_sequences: seq_length = {seq_length}")
    logger.info(f"[DEBUG] create_sequences: target_col_idx = {target_col_idx}")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col_idx])
    logger.info(f"[DEBUG] create_sequences: len(X) = {len(X)}")
    logger.info(f"[DEBUG] create_sequences: len(y) = {len(y)}")
    return np.array(X), np.array(y)

def prepare_data_for_model(df: pd.DataFrame, seq_length: int, test_size: float = 0.2) -> Tuple:
    """Prepare data for the model, handling datetime columns properly."""
    logger.info(f"[DEBUG] prepare_data_for_model: df.shape = {df.shape}")
    # Create a copy of the dataframe
    df = df.copy()
    
    # Store the datetime column separately
    datetime_col = df['date']
    
    # Get numeric columns (excluding datetime)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Scale the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Find the index of the 'close' column
    close_col_idx = list(numeric_cols).index('close')
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length, close_col_idx)
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Get corresponding dates for test set
    # The first seq_length dates are used for the first sequence, so we need to offset by that
    # Also need to ensure we only get dates for the test set predictions
    test_dates = datetime_col[train_size + seq_length:].values
    # test_dates = datetime_col[seq_length + train_size:seq_length + train_size + len(X_test)].values
    
    return X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols

def build_cnn_lstm_model(trial, seq_length, n_features, **kwargs):
    """CNN+LSTM model with variable number of LSTM layers."""
    # Get hyperparameters from kwargs
    cnn_filters = kwargs.get('cnn_filters', 64)
    cnn_kernel = kwargs.get('cnn_kernel', 3)
    cnn_dropout = kwargs.get('cnn_dropout', 0.2)
    lstm_layers = kwargs.get('lstm_layers', 2)
    lstm_units = kwargs.get('lstm_units', 64)
    lstm_dropout = kwargs.get('lstm_dropout', 0.2)
    learning_rate = kwargs.get('learning_rate', 0.001)
    
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))
    model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(cnn_dropout))
    
    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_seq))
        model.add(Dropout(lstm_dropout))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_lstm_model(trial, seq_length, n_features, **kwargs):
    """LSTM model with variable number of LSTM layers."""
    # Get hyperparameters from kwargs
    lstm_layers = kwargs.get('lstm_layers', 2)
    lstm_units = kwargs.get('lstm_units', 64)
    dropout_rate = kwargs.get('dropout_rate', 0.2)
    learning_rate = kwargs.get('learning_rate', 0.001)
    dense_units = kwargs.get('dense_units', 32)

    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        units = lstm_units if i == 0 else lstm_units // 2
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def calculate_metrics(predictions: np.ndarray, actual: np.ndarray, threshold: float = 0.005) -> Dict[str, float]:
    """
    Calculate various evaluation metrics for the model predictions.
    
    Args:
        predictions: Array of predicted prices
        actual: Array of actual prices
        threshold: Threshold for Directional Change events (default: 0.5%)
    
    Returns:
        Dictionary containing:
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - directional_accuracy: Accuracy of up/down predictions
        - dc_precision: Precision of Directional Change events
        - dc_recall: Recall of Directional Change events
        - dc_timing_error: Average timing error for DC events
    """
    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    
    # Calculate directional accuracy
    pred_changes = np.diff(predictions)
    actual_changes = np.diff(actual)
    directional_accuracy = np.mean(np.sign(pred_changes) == np.sign(actual_changes))
    
    # Calculate Directional Change (DC) metrics
    actual_dc = np.abs(np.diff(actual) / actual[:-1]) >= threshold
    pred_dc = np.abs(np.diff(predictions) / predictions[:-1]) >= threshold
    
    # Calculate DC precision and recall
    true_positives = np.sum(actual_dc & pred_dc)
    false_positives = np.sum(~actual_dc & pred_dc)
    false_negatives = np.sum(actual_dc & ~pred_dc)
    
    dc_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    dc_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate timing error for DC events
    dc_timing_error = 0
    if np.sum(actual_dc) > 0:
        dc_indices = np.where(actual_dc)[0]
        timing_errors = []
        for idx in dc_indices:
            # Find the closest predicted DC event within a window
            window = 5  # Look 5 days before and after
            pred_dc_in_window = np.where(pred_dc[max(0, idx-window):min(len(pred_dc), idx+window)])[0]
            if len(pred_dc_in_window) > 0:
                closest_pred = pred_dc_in_window[np.argmin(np.abs(pred_dc_in_window - window))]
                timing_errors.append(abs(closest_pred - window))
        dc_timing_error = np.mean(timing_errors) if timing_errors else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'directional_accuracy': directional_accuracy,
        'dc_precision': dc_precision,
        'dc_recall': dc_recall,
        'dc_timing_error': dc_timing_error
    }

def train_model(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    seq_length: int,
    n_trials: int = 20,
    build_model_fn: Callable = build_lstm_model,
    model_params: Dict[str, Any] = None,
    half_life_range: Union[tuple, float] = (0.5, 10.0),
    n_days_range: Union[tuple, int] = (1, 10),
    use_cross_validation: bool = False,
    n_splits: int = 5,
    use_learning_rate_scheduler: bool = False,
    use_regularization: bool = False,
    lstm_layers_range: Union[tuple, int] = (2, 5),
    lstm_units_range: Union[tuple, int] = (32, 128),
    dropout_rate_range: Union[tuple, float] = (0.1, 0.5),
    learning_rate_range: Union[tuple, float] = (1e-5, 1e-2),
    dense_units_range: Union[tuple, int] = (16, 64),
    cnn_filters_range: Union[tuple, int] = (32, 128),
    cnn_kernel_range: Union[tuple, int] = (2, 5),
    cnn_dropout_range: Union[tuple, float] = (0.1, 0.5),
    storage_name: str = None,
    use_existing_storage: bool = False
) -> Tuple:
    """
    Train and evaluate the model with hyperparameter optimization.
    
    Args:
        news_df: DataFrame containing news sentiment data
        price_df: DataFrame with price data ['date', 'close', 'volume']
        seq_length: Length of input sequences for the model
        n_trials: Number of Optuna trials for hyperparameter optimization
        build_model_fn: Function to build the model (LSTM or CNN-LSTM)
        model_params: Additional parameters for the model
        half_life_range: Range for half-life days parameter in news aggregation
        n_days_range: Range for number of days parameter in news aggregation
        use_cross_validation: Whether to use k-fold cross-validation
        n_splits: Number of splits for cross-validation
        use_learning_rate_scheduler: Whether to use learning rate scheduling
        use_regularization: Whether to use L1/L2 regularization
        lstm_layers_range: Range for number of LSTM layers
        lstm_units_range: Range for number of LSTM units
        dropout_rate_range: Range for dropout rate
        learning_rate_range: Range for learning rate
        dense_units_range: Range for number of dense units
        cnn_filters_range: Range for number of CNN filters
        cnn_kernel_range: Range for CNN kernel size
        cnn_dropout_range: Range for CNN dropout rate
        storage_name: Name of the Optuna storage database
        use_existing_storage: Whether to use existing Optuna storage
    
    Returns:
        Tuple containing:
        - best_model: Trained model with best hyperparameters
        - history: Training history
        - predictions: Model predictions on test set
        - actual: Actual values from test set
        - test_dates: Dates corresponding to test set
        - metrics: Dictionary containing evaluation metrics
        - best_params: Dictionary of best hyperparameters
    """
    start_time = time.time()
    logger.info("Starting model training process...")

    def objective(trial):
        trial_start_time = time.time()
        logger.info(f"Starting trial {trial.number + 1}/{n_trials}")

        # 1. Get hyperparameters for this trial
        half_life_days = half_life_range if isinstance(half_life_range, float) else trial.suggest_float('half_life_days', half_life_range[0], half_life_range[1])
        n_days = n_days_range if isinstance(n_days_range, int) else trial.suggest_int('n_days', n_days_range[0], n_days_range[1])
        # Use the provided seq_length directly
        current_seq_length = seq_length
        
        # Model-specific hyperparameters
        if build_model_fn == build_lstm_model:
            lstm_layers = lstm_layers_range if isinstance(lstm_layers_range, int) else trial.suggest_int('lstm_layers', lstm_layers_range[0], lstm_layers_range[1])
            lstm_units = lstm_units_range if isinstance(lstm_units_range, int) else trial.suggest_int('lstm_units', lstm_units_range[0], lstm_units_range[1])
            dropout_rate = dropout_rate_range if isinstance(dropout_rate_range, float) else trial.suggest_float('dropout_rate', dropout_rate_range[0], dropout_rate_range[1])
            learning_rate = learning_rate_range if isinstance(learning_rate_range, float) else trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1], log=True)
            dense_units = dense_units_range if isinstance(dense_units_range, int) else trial.suggest_int('dense_units', dense_units_range[0], dense_units_range[1])
            
            # Pass hyperparameters to model building function
            model_params = {
                'lstm_layers': lstm_layers,
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'dense_units': dense_units
            }
        else:  # CNN-LSTM model
            cnn_filters = cnn_filters_range if isinstance(cnn_filters_range, int) else trial.suggest_int('cnn_filters', cnn_filters_range[0], cnn_filters_range[1])
            cnn_kernel = cnn_kernel_range if isinstance(cnn_kernel_range, int) else trial.suggest_int('cnn_kernel', cnn_kernel_range[0], cnn_kernel_range[1])
            cnn_dropout = cnn_dropout_range if isinstance(cnn_dropout_range, float) else trial.suggest_float('cnn_dropout', cnn_dropout_range[0], cnn_dropout_range[1])
            lstm_layers = lstm_layers_range if isinstance(lstm_layers_range, int) else trial.suggest_int('lstm_layers', lstm_layers_range[0], lstm_layers_range[1])
            lstm_units = lstm_units_range if isinstance(lstm_units_range, int) else trial.suggest_int('lstm_units', lstm_units_range[0], lstm_units_range[1])
            lstm_dropout = dropout_rate_range if isinstance(dropout_rate_range, float) else trial.suggest_float('lstm_dropout', dropout_rate_range[0], dropout_rate_range[1])
            learning_rate = learning_rate_range if isinstance(learning_rate_range, float) else trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1], log=True)
            
            # Pass hyperparameters to model building function
            model_params = {
                'cnn_filters': cnn_filters,
                'cnn_kernel': cnn_kernel,
                'cnn_dropout': cnn_dropout,
                'lstm_layers': lstm_layers,
                'lstm_units': lstm_units,
                'lstm_dropout': lstm_dropout,
                'learning_rate': learning_rate
            }

        # 2. Aggregate data using the suggested half_life_days
        if 'sentiment_score_finbert' in news_df.columns:
            logger.info("Detected FinBERT data format - using FinBERT aggregation")
            aggregated_df = aggregate_news_price_rolling_finbert(
                news_df=news_df,
                price_df=price_df,
                n_days=n_days,
                half_life_days=half_life_days
            )
        elif 'event_type' in news_df.columns:
            logger.info("Detected LLM data with event types - using LLM aggregation")
            aggregated_df = aggregate_news_price_rolling_llm(
                news_df=news_df,
                price_df=price_df,
                event_types=EVENT_TYPES,
                n_days=n_days,
                half_life_days=half_life_days
            )
        elif 'sentiment_score_llm' in news_df.columns:
            logger.info("Detected LLM sentiment data - using LLM sentiment aggregation")
            aggregated_df = aggregate_news_price_rolling_llm_sentiment(
                news_df=news_df,
                price_df=price_df,
                n_days=n_days,
                half_life_days=half_life_days
            )
        else:
            raise ValueError("Input data must contain either 'sentiment_score_finbert', 'sentiment_score_llm', or 'event_type' column")

        # 3. Prepare data for the model
        X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
            aggregated_df, seq_length=current_seq_length
        )

        # 4. Set up cross-validation if enabled
        if use_cross_validation:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                
                # Build and train the model
                model = build_model_fn(trial, current_seq_length, X_train.shape[2], **model_params)
                
                # Add learning rate scheduler if enabled
                callbacks = []
                if use_learning_rate_scheduler:
                    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                    callbacks.append(lr_scheduler)
                
                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                
                # Train the model
                history = model.fit(
                    X_train_cv, y_train_cv,
                    validation_data=(X_val_cv, y_val_cv),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                cv_scores.append(min(history.history['val_loss']))
            
            # Return mean validation loss across folds
            trial_time = time.time() - trial_start_time
            logger.info(f"Completed trial {trial.number + 1}/{n_trials} in {trial_time:.2f} seconds")
            return np.mean(cv_scores)
        
        else:
            # 5. Split training data into train and validation sets
            val_size = int(len(X_train) * 0.2)
            X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

            # 6. Build and train the model
            model = build_model_fn(trial, current_seq_length, X_train.shape[2], **model_params)
            
            # Add learning rate scheduler if enabled
            callbacks = []
            if use_learning_rate_scheduler:
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
                callbacks.append(lr_scheduler)
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # Train the model
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            # 7. Return the best validation loss
            trial_time = time.time() - trial_start_time
            logger.info(f"Completed trial {trial.number + 1}/{n_trials} in {trial_time:.2f} seconds")
            return min(history.history['val_loss'])

    # 8. Run Optuna optimization
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    # Get job and sub-job names from environment variables
    job_name = os.getenv('JOB_NAME', 'default_job')
    sub_job_name = os.getenv('SUB_JOB_NAME', 'default_sub_job')
    
    # Set up Optuna storage
    study = get_study(job_name, sub_job_name, use_existing=use_existing_storage)
    
    study.optimize(objective, n_trials=n_trials)
    optuna_time = time.time() - start_time
    logger.info(f"Completed hyperparameter optimization in {optuna_time:.2f} seconds")

    # 9. After finding the best trial, rebuild the dataset with the best hyperparameters
    best_params = study.best_params
    best_half_life = best_params.get('half_life_days', half_life_range if isinstance(half_life_range, float) else half_life_range[0])
    best_n_days = best_params.get('n_days', n_days_range if isinstance(n_days_range, int) else n_days_range[0])
    # Use the provided seq_length directly
    best_seq_length = seq_length
    
    # Remove seq_length from best_params to avoid duplicate argument
    if 'seq_length' in best_params:
        del best_params['seq_length']
    
    # Use the appropriate aggregation function based on the data type
    if 'sentiment_score_finbert' in news_df.columns:
        best_aggregated_df = aggregate_news_price_rolling_finbert(
            news_df=news_df,
            price_df=price_df,
            n_days=best_n_days,
            half_life_days=best_half_life
        )
    elif 'event_type' in news_df.columns:
        best_aggregated_df = aggregate_news_price_rolling_llm(
            news_df=news_df,
            price_df=price_df,
            event_types=EVENT_TYPES,
            n_days=best_n_days,
            half_life_days=best_half_life
        )
    elif 'sentiment_score_llm' in news_df.columns:
        best_aggregated_df = aggregate_news_price_rolling_llm_sentiment(
            news_df=news_df,
            price_df=price_df,
            n_days=best_n_days,
            half_life_days=best_half_life
        )
    else:
        raise ValueError("Input data must contain either 'sentiment_score_finbert', 'sentiment_score_llm', or 'event_type' column")

    # 10. Prepare data for the final model
    X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
        best_aggregated_df, seq_length=best_seq_length
    )

    # 11. Train the final model with the best hyperparameters
    logger.info("Training final model with best hyperparameters...")
    final_model_start_time = time.time()
    best_model = build_model_fn(study.best_trial, best_seq_length, X_train.shape[2], **best_params)
    
    # Add learning rate scheduler if enabled
    callbacks = []
    if use_learning_rate_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        callbacks.append(lr_scheduler)
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    final_model_time = time.time() - final_model_start_time
    logger.info(f"Completed final model training in {final_model_time:.2f} seconds")

    # 12. Make predictions
    logger.info("Making predictions on test set...")
    predictions = best_model.predict(X_test)
    close_col_idx = list(numeric_cols).index('close')
    pred_reshaped = np.zeros((len(predictions), len(numeric_cols)))
    pred_reshaped[:, close_col_idx] = predictions.flatten()
    predictions = scaler.inverse_transform(pred_reshaped)[:, close_col_idx]
    actual_reshaped = np.zeros((len(y_test), len(numeric_cols)))
    actual_reshaped[:, close_col_idx] = y_test
    actual = scaler.inverse_transform(actual_reshaped)[:, close_col_idx]

    # 13. Calculate metrics
    metrics = calculate_metrics(predictions, actual)
    
    # Log metrics
    logger.info("\nModel Evaluation Metrics:")
    logger.info(f"RMSE = ${metrics['RMSE']:.2f}, MAE = ${metrics['MAE']:.2f} on next-day close")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    logger.info(f"DC Precision: {metrics['dc_precision']:.2%}")
    logger.info(f"DC Recall: {metrics['dc_recall']:.2%}")
    logger.info(f"DC Timing Error: {metrics['dc_timing_error']:.2f} days")

    total_time = time.time() - start_time
    logger.info(f"Total training process completed in {total_time:.2f} seconds")

    # Save the best model with sub-job name
    model_save_name = f'saved_models/{job_name}_{sub_job_name}_best.keras'
    best_model.save(model_save_name)
    logger.info(f"Saved best model to {model_save_name}")

    return best_model, history, predictions, actual, test_dates, metrics, best_params

def plot_results(predictions_list, line_names, actual, test_dates, metrics_list=None, job_name=None):
    """Plot the results and save the plot to a file."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15), height_ratios=[2, 1])
    
    # Plot actual price and predictions
    ax1.plot(test_dates, actual, label='Actual', color='blue')
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    for pred, name, color in zip(predictions_list, line_names, colors):
        ax1.plot(test_dates, pred, label=name, color=color)
    
    ax1.set_title('Actual vs Predicted Stock Prices')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot directional accuracy if metrics are provided
    if metrics_list:
        model_names = line_names
        metrics = ['RMSE', 'MAE', 'directional_accuracy', 'dc_precision', 'dc_recall']
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [m[metric] for m in metrics_list]
            ax2.bar(x + i*width, values, width, label=metric)
        
        ax2.set_ylabel('Score')
        ax2.set_title('Model Comparison Metrics')
        ax2.set_xticks(x + width*2)
        ax2.set_xticklabels(model_names)
        ax2.legend()
    
    # Save the plot with consistent start time
    os.makedirs('plots', exist_ok=True)
    start_time = os.environ.get('START_TIME', datetime.now().strftime("%Y%m%d_%H%M%S"))
    plot_filename = f'plots/{job_name}_{start_time}.png'
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def save_results_to_csv(results: dict, filename: str):
    """
    Save the results dictionary to a CSV file. Each row contains sub-job name, date, prediction, actual, and metrics.
    """
    # Collect all sub-job names
    sub_job_names = list(results.keys())
    # Open CSV file for writing
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['sub_job', 'date', 'prediction', 'actual']
        # Add metrics as separate columns (use the first sub-job's metrics as reference)
        if sub_job_names:
            metrics_keys = list(results[sub_job_names[0]]['metrics'].keys())
            fieldnames.extend(metrics_keys)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Write rows for each sub-job
        for sub_job in sub_job_names:
            preds = results[sub_job]['predictions']
            actuals = results[sub_job]['actual']
            dates = results[sub_job]['test_dates']
            metrics = results[sub_job]['metrics']
            # Write one row per prediction
            for i in range(len(preds)):
                row = {
                    'sub_job': sub_job,
                    'date': dates[i] if i < len(dates) else '',
                    'prediction': preds[i],
                    'actual': actuals[i] if i < len(actuals) else ''
                }
                # Add metrics (same for all rows in this sub-job)
                for k in metrics.keys():
                    row[k] = metrics[k]
                writer.writerow(row) 