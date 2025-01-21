from typing import Dict, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def evaluate_imputation(original: pd.DataFrame, imputed: pd.DataFrame,
                        columns: List[str], method_name: str) -> Dict:
    """
    Compare summary statistics (mean, median, variance) between original and imputed data.

    Args:
        original: Original DataFrame with missing values.
        imputed: Imputed DataFrame.
        columns: Columns to evaluate.
        method_name: Name of the imputation method.

    Returns:
        Dict: Dictionary containing summary statistics comparison.
    """
    try:
        metrics = {}
        for col in columns:
            # Compute statistics for original data (excluding missing values)
            original_stats = {
                'mean': round(original[col].mean(skipna=True), 5),
                'median': round(original[col].median(skipna=True), 5),
                'variance': round(original[col].var(skipna=True), 5)
            }

            # Compute statistics for imputed data
            imputed_stats = {
                'mean': round(imputed[col].mean(), 5),
                'median': round(imputed[col].median(), 5),
                'variance': round(imputed[col].var(), 5)
            }

            # Calculate the differences between original and imputed statistics
            differences = {
                'mean_diff': round(abs(imputed_stats['mean'] - original_stats['mean']), 5),
                'median_diff': round(abs(imputed_stats['median'] - original_stats['median']), 5),
                'variance_diff': round(abs(imputed_stats['variance'] - original_stats['variance']), 5)
            }

            metrics[col] = {
                'original': original_stats,
                'imputed': imputed_stats,
                'differences': differences
            }

        logger.info(f"Summary statistics comparison for {method_name}")
        logger.info(metrics)
        logger.info(f"Completed evaluation for {method_name}")
        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

