import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class ImputationVisualizer:
    """Visualization utilities for imputation results."""

    @staticmethod
    def plot_distributions(original: pd.DataFrame, imputed: pd.DataFrame,
                           columns: List[str], method: str, output_dir: str = "./visualization/pictures/forecasted_Custom_Imputation"):
        """
        Plot distribution comparison between original and imputed data and save plots.

        Parameters:
            original (pd.DataFrame): Original dataset.
            imputed (pd.DataFrame): Imputed dataset.
            columns (List[str]): List of column names to plot.
            method (str): Name of the imputation method.
            output_dir (str): Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)

        for col in columns:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=original[col].dropna(), label='Original', alpha=0.5)
            sns.kdeplot(data=imputed[col], label=f'Imputed ({method})', alpha=0.5)
            plt.title(f'Distribution Comparison - {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()

            # Save plot as PNG
            filename = f"{output_dir}/{col}_{method}_imputation.png"
            plt.savefig(filename, format='png')
            logger.info(f"Saved plot: {filename}")

            plt.close()

    @staticmethod
    def plot_missing_patterns(data: pd.DataFrame, columns: List[str], output_dir: str = "/visualization/pictures/forecasted_Custom_Imputation/missing_Patterns"):
        """
        Plot missing data patterns and save the plot.

        Parameters:
            data (pd.DataFrame): Dataset to analyze.
            columns (List[str]): List of column names to plot.
            output_dir (str): Directory to save the plot.
        """
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.heatmap(data[columns].isna(), cmap='viridis', cbar_kws={'label': 'Missing'})
        plt.title('Missing Data Patterns')

        # Save plot as PNG
        filename = f"{output_dir}/missing_patterns.png"
        plt.savefig(filename, format='png')
        logger.info(f"Saved plot: {filename}")

        plt.close()

    @staticmethod
    def plot_evaluation_metrics(evaluations: Dict, output_dir: str = "/visualization/pictures/forecasted_Custom_Imputation/"):
        """
        Plot evaluation metrics comparison and save plots.

        Parameters:
            evaluations (Dict): Dictionary of evaluation metrics.
            output_dir (str): Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)

        metrics_df = pd.DataFrame([
            {
                'method': method,
                'column': col,
                'metric': metric,
                'value': value
            }
            for method, method_results in evaluations.items()
            for col, col_results in method_results.items()
            for metric, value in col_results.items()
        ])

        g = sns.FacetGrid(metrics_df, col='metric', row='column', height=4)
        g.map_dataframe(sns.barplot, x='method', y='value')
        g.fig.suptitle('Imputation Method Comparison', y=1.02)

        filename = f"{output_dir}/evaluation_metrics_comparison.png"
        g.savefig(filename, format='png')
        logger.info(f"Saved plot: {filename}")

        plt.close()
