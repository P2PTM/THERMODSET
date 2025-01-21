from typing import List
import pandas as pd
from .base import BaseImputer
import logging

logger = logging.getLogger(__name__)


class InterpolationImputer(BaseImputer):
    """Interpolation-based imputation methods."""

    def impute(self, data: pd.DataFrame, columns: List[str], method: str = 'linear') -> pd.DataFrame:
        """
        Perform interpolation using specified method.

        Args:
            data: Input DataFrame
            columns: Columns to impute
            method: Interpolation method ('linear', 'spline', 'time')

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        imputed = data.copy()

        try:
            for col in columns:
                if method == 'spline':
                    imputed[col] = imputed[col].interpolate(method='spline', order=3)
                else:
                    imputed[col] = imputed[col].interpolate(method=method)

            logger.info(f"Completed {method} interpolation")
            return imputed

        except Exception as e:
            logger.error(f"Error during {method} interpolation: {e}")
            raise
