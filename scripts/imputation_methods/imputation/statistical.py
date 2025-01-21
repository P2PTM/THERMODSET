from typing import List
import pandas as pd
import numpy as np
from .base import BaseImputer
import logging

logger = logging.getLogger(__name__)


class StatisticalImputer(BaseImputer):
    """Statistical imputation methods."""

    def impute(self, data: pd.DataFrame, columns: List[str], method: str = 'mean') -> pd.DataFrame:
        """
        Perform statistical imputation using specified method.

        Args:
            data: Input DataFrame
            columns: Columns to impute
            method: Imputation method ('mean', 'median', 'locf', 'nocb')

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        imputed = data.copy()

        try:
            if method == 'mean':
                for col in columns:
                    imputed[col].fillna(imputed[col].mean(), inplace=True)
            elif method == 'median':
                for col in columns:
                    imputed[col].fillna(imputed[col].median(), inplace=True)
            elif method == 'locf':
                imputed[columns] = imputed[columns].fillna(method='ffill')
            elif method == 'nocb':
                imputed[columns] = imputed[columns].fillna(method='bfill')
            else:
                raise ValueError(f"Unknown method: {method}")

            logger.info(f"Completed {method} imputation")
            return imputed

        except Exception as e:
            logger.error(f"Error during {method} imputation: {e}")
            raise