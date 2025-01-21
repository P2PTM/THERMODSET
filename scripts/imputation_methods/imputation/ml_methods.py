from typing import List
import pandas as pd
from sklearn.impute import KNNImputer
from statsmodels.tsa.arima.model import ARIMA
from .base import BaseImputer
import logging

logger = logging.getLogger(__name__)


class MLImputer(BaseImputer):
    """Machine learning-based imputation methods."""

    def __init__(self, n_neighbors: int = 60, arima_order: tuple = (1, 1, 1)):
        self.n_neighbors = n_neighbors
        self.arima_order = arima_order

    def impute(self, data: pd.DataFrame, columns: List[str], method: str = 'knn') -> pd.DataFrame:
        """
        Perform ML-based imputation.

        Args:
            data: Input DataFrame
            columns: Columns to impute
            method: Imputation method ('knn', 'arima')

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        imputed = data.copy()

        try:
            if method == 'knn':
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                imputed[columns] = imputer.fit_transform(imputed[columns])
            elif method == 'arima':
                for col in columns:
                    imputed[col] = self._arima_imputation(imputed[col])
            else:
                raise ValueError(f"Unknown method: {method}")

            logger.info(f"Completed {method} imputation")
            return imputed

        except Exception as e:
            logger.error(f"Error during {method} imputation: {e}")
            raise

    def _arima_imputation(self, series: pd.Series) -> pd.Series:
        """Helper method for ARIMA imputation."""
        series_imputed = series.copy()
        for i in range(len(series)):
            if pd.isna(series[i]):
                model = ARIMA(series[:i].dropna(), order=self.arima_order)
                model_fit = model.fit()
                series_imputed[i] = model_fit.forecast()[0]
        return series_imputed
