from typing import List

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_data(data: pd.DataFrame, columns: List[str]) -> bool:
    """Validate input data format and contents."""
    try:
        # Check if required columns exist
        missing_cols = set(columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data types
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")

        # Check for infinite values
        if data[columns].isin([np.inf, -np.inf]).any().any():
            raise ValueError("Data contains infinite values")

        return True

    except Exception as e:
        logger.error(f"Data validation error: {e}")
        raise