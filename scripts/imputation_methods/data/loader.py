import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and preprocessing operations."""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def preprocess_data(data: pd.DataFrame,
                        columns_with_nan: List[str],
                        invalid_value: int = -1) -> pd.DataFrame:
        """
        Preprocess data by replacing invalid (-1) values with NaN.

        Args:
            data: Input DataFrame
            columns_with_nan: Columns to process
            invalid_value: Value to replace with NaN

        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            data = data.copy()
            data[columns_with_nan] = data[columns_with_nan].replace(invalid_value, np.nan)
            logger.info(f"Preprocessing completed. NaN counts: {data[columns_with_nan].isna().sum()}")
            return data
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
