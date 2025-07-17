import os
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
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

logger = get_logger(__name__)

class OptunaResultsAnalyzer:
    def __init__(self, output_dir: str = "optuna_analysis"):
        """Initialize the analyzer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.studies = {}
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

    def get_study_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all studies."""
        summary_data = []
        
        for name, study in self.studies.items():
            if len(study.trials) == 0:
                continue
                
            best_trial = study.best_trial
            summary_data.append({
                'study_name': name,
                'n_trials': len(study.trials),
                'best_value': best_trial.value,
                'best_params': json.dumps(best_trial.params),
                'datetime_start': study.trials[0].datetime_start,
                'datetime_complete': study.trials[-1].datetime_complete,
                'duration': (study.trials[-1].datetime_complete - study.trials[0].datetime_start).total_seconds() / 3600  # hours
            })
            
        return pd.DataFrame(summary_data)

    def plot_optimization_history(self, study_name: str):
        """Plot optimization history for a specific study."""
        study = self.studies[study_name]
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add optimization history
        values = [t.value for t in study.trials if t.value is not None]
        trials = list(range(len(values)))
        
        fig.add_trace(
            go.Scatter(x=trials, y=values, name="Objective Value"),
            secondary_y=False,
        )
        
        # Add best value line
        best_values = [min(values[:i+1]) for i in range(len(values))]
        fig.add_trace(
            go.Scatter(x=trials, y=best_values, name="Best Value", line=dict(dash='dash')),
            secondary_y=False,
        )
        
        # Add parameter values
        for param_name in study.best_params.keys():
            param_values = [t.params.get(param_name) for t in study.trials if t.value is not None]
            fig.add_trace(
                go.Scatter(x=trials, y=param_values, name=param_name),
                secondary_y=True,
            )
        
        fig.update_layout(
            title=f"Optimization History - {study_name}",
            xaxis_title="Trial",
            yaxis_title="Objective Value",
            yaxis2_title="Parameter Value",
            showlegend=True
        )
        
        fig.write_html(os.path.join(self.output_dir, f"{study_name}_optimization_history.html"))
        fig.write_image(os.path.join(self.output_dir, f"{study_name}_optimization_history.png"))

    def plot_parameter_importance(self, study_name: str):
        """Plot parameter importance for a specific study."""
        study = self.studies[study_name]
        
        try:
            importance = optuna.importance.get_param_importances(study)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    text=[f"{v:.3f}" for v in importance.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"Parameter Importance - {study_name}",
                xaxis_title="Parameter",
                yaxis_title="Importance Score",
                showlegend=False
            )
            
            fig.write_html(os.path.join(self.output_dir, f"{study_name}_parameter_importance.html"))
            fig.write_image(os.path.join(self.output_dir, f"{study_name}_parameter_importance.png"))
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance for {study_name}: {e}")

    def plot_parameter_distributions(self, study_name: str):
        """Plot parameter distributions for a specific study."""
        study = self.studies[study_name]
        
        # Get all parameter names
        param_names = list(study.best_params.keys())
        n_params = len(param_names)
        
        # Create subplots
        fig = make_subplots(rows=n_params, cols=1, subplot_titles=param_names)
        
        for i, param_name in enumerate(param_names, 1):
            # Get parameter values and corresponding objective values
            param_values = []
            objective_values = []
            
            for trial in study.trials:
                if trial.value is not None and param_name in trial.params:
                    param_values.append(trial.params[param_name])
                    objective_values.append(trial.value)
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=param_values,
                    y=objective_values,
                    mode='markers',
                    name=param_name,
                    marker=dict(
                        size=8,
                        color=objective_values,
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=i, col=1
            )
            
            fig.update_xaxes(title_text=param_name, row=i, col=1)
            fig.update_yaxes(title_text="Objective Value", row=i, col=1)
        
        fig.update_layout(
            title=f"Parameter Distributions - {study_name}",
            height=300 * n_params,
            showlegend=False
        )
        
        fig.write_html(os.path.join(self.output_dir, f"{study_name}_parameter_distributions.html"))
        fig.write_image(os.path.join(self.output_dir, f"{study_name}_parameter_distributions.png"))

    def generate_summary_report(self):
        """Generate a markdown summary report of all studies."""
        summary_df = self.get_study_summary()
        
        report = f"""# Optuna Study Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Study Summary
{summary_df.to_markdown()}

## Best Parameters by Study
"""
        
        for name, study in self.studies.items():
            if len(study.trials) == 0:
                continue
                
            report += f"\n### {name}\n"
            report += f"Best Value: {study.best_value:.4f}\n"
            report += "Best Parameters:\n"
            for param, value in study.best_params.items():
                report += f"- {param}: {value}\n"
            report += "\n"
        
        # Save report
        with open(os.path.join(self.output_dir, "summary_report.md"), "w") as f:
            f.write(report)

    def analyze_all_studies(self):
        """Run all analysis for all studies."""
        # Generate summary report
        self.generate_summary_report()
        
        # Generate plots for each study
        for name in self.studies.keys():
            logger.info(f"Analyzing study: {name}")
            self.plot_optimization_history(name)
            self.plot_parameter_importance(name)
            self.plot_parameter_distributions(name)

def main():
    """Main function to run the analysis."""
    analyzer = OptunaResultsAnalyzer()
    analyzer.analyze_all_studies()
    logger.info("Analysis complete. Results saved in optuna_analysis directory.")

if __name__ == "__main__":
    main() 