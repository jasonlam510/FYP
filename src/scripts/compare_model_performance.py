import os
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

from src.utils.optuna_storage import get_all_studies, STORAGE_URL
from src.utils.logger import get_logger
from src.data.news import NewsData
from src.data.price import PriceData
from src.data.mi import MIData
from src.model.train_model import train_model, calculate_metrics
from src.utils.combine import combine_mi_price, aggregate_news_price_rolling_llm
from src.helper.ti import calculate_technical_indicators
from src.helper.dc import add_dc_event_features
from src.config import (
    SEQ_LENGTH,
    BALANCED_INDICATORS,
    HALF_LIFE_RANGE,
    N_DAYS_RANGE
)

logger = get_logger(__name__)

class ModelPerformanceAnalyzer:
    def __init__(self, output_dir: str = "model_comparison"):
        """Initialize the analyzer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.studies = {}
        self.results = {}
        self.load_studies()
        
    def load_studies(self):
        """Load all studies from the storage."""
        study_names = get_all_studies()
        for name in study_names:
            try:
                study = optuna.load_study(study_name=name, storage=STORAGE_URL)
                self.studies[name] = study
                logger.info(f"Loaded study: {name}")
            except Exception as e:
                logger.error(f"Error loading study {name}: {e}")

    def prepare_data(self):
        """Prepare data for model evaluation."""
        logger.info("Loading and preparing data...")
        macro_data = MIData()
        news_data = NewsData()
        price_data = PriceData()

        # Get datasets
        self.llm_df = news_data.get_llm_df()
        self.price_df = price_data.get_price_df()
        self.macro_df = macro_data.get_mi_df()

        # Process price data
        logger.info("Processing price data...")
        self.price_df = calculate_technical_indicators(self.price_df, BALANCED_INDICATORS)
        self.price_dc_df = add_dc_event_features(self.price_df)
        self.processed_price_df = combine_mi_price(self.macro_df, self.price_dc_df)

    def evaluate_models(self):
        """Evaluate all models using their best parameters."""
        self.prepare_data()
        
        for study_name, study in self.studies.items():
            if len(study.trials) == 0:
                continue
                
            logger.info(f"\nEvaluating model from study: {study_name}")
            
            # Get best parameters
            best_params = study.best_params
            
            # Determine model type from study name
            if 'cnn' in study_name.lower():
                from src.model.train_model import build_cnn_lstm_model as build_model_fn
            else:
                from src.model.train_model import build_lstm_model as build_model_fn
            
            # Train model with best parameters
            model, history, pred, actual, test_dates, metrics, params = train_model(
                news_df=self.llm_df,
                price_df=self.processed_price_df,
                seq_length=SEQ_LENGTH,
                n_trials=1,  # Only one trial since we're using best params
                build_model_fn=build_model_fn,
                model_params=best_params,
                half_life_range=HALF_LIFE_RANGE,
                n_days_range=N_DAYS_RANGE,
                use_existing_storage=False  # Don't use storage for evaluation
            )
            
            self.results[study_name] = {
                'predictions': pred,
                'actual': actual,
                'test_dates': test_dates,
                'metrics': metrics,
                'best_params': best_params
            }

    def plot_metrics_comparison(self):
        """Create comparison plots of metrics across all models."""
        # Prepare data for plotting
        metrics_data = []
        for study_name, result in self.results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': study_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'Directional Accuracy': metrics['directional_accuracy'],
                'DC Precision': metrics['dc_precision'],
                'DC Recall': metrics['dc_recall'],
                'DC Timing Error': metrics['dc_timing_error']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create radar chart for accuracy metrics
        accuracy_metrics = ['Directional Accuracy', 'DC Precision', 'DC Recall']
        fig = go.Figure()
        
        for _, row in metrics_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in accuracy_metrics],
                theta=accuracy_metrics,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Accuracy Metrics Comparison"
        )
        
        fig.write_html(os.path.join(self.output_dir, "accuracy_metrics_radar.html"))
        fig.write_image(os.path.join(self.output_dir, "accuracy_metrics_radar.png"))
        
        # Create bar chart for error metrics
        error_metrics = ['RMSE', 'MAE', 'DC Timing Error']
        fig = go.Figure()
        
        for metric in error_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=[f"{v:.3f}" for v in metrics_df[metric]],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Error Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Error Value",
            showlegend=True
        )
        
        fig.write_html(os.path.join(self.output_dir, "error_metrics_bar.html"))
        fig.write_image(os.path.join(self.output_dir, "error_metrics_bar.png"))

    def generate_summary_report(self):
        """Generate a markdown summary report of model performance."""
        report = f"""# Model Performance Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics Summary
"""
        
        # Create metrics table
        metrics_data = []
        for study_name, result in self.results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': study_name,
                'RMSE': f"${metrics['RMSE']:.2f}",
                'MAE': f"${metrics['MAE']:.2f}",
                'Directional Accuracy': f"{metrics['directional_accuracy']:.2%}",
                'DC Precision': f"{metrics['dc_precision']:.2%}",
                'DC Recall': f"{metrics['dc_recall']:.2%}",
                'DC Timing Error': f"{metrics['dc_timing_error']:.2f} days"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        report += metrics_df.to_markdown(index=False)
        
        # Add best parameters for each model
        report += "\n\n## Best Parameters by Model\n"
        for study_name, result in self.results.items():
            report += f"\n### {study_name}\n"
            for param, value in result['best_params'].items():
                report += f"- {param}: {value}\n"
        
        # Save report
        with open(os.path.join(self.output_dir, "performance_report.md"), "w") as f:
            f.write(report)

    def analyze_all_models(self):
        """Run all analysis for all models."""
        self.evaluate_models()
        self.plot_metrics_comparison()
        self.generate_summary_report()
        logger.info("Analysis complete. Results saved in model_comparison directory.")

def main():
    """Main function to run the analysis."""
    analyzer = ModelPerformanceAnalyzer()
    analyzer.analyze_all_models()

if __name__ == "__main__":
    main() 