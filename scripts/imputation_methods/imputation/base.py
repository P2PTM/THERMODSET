from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class BaseImputer(ABC):
    """Base class for imputation methods."""

    @abstractmethod
    def impute(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute missing values in specified columns.

        Args:
            data: Input DataFrame
            columns: Columns to impute

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        pass
