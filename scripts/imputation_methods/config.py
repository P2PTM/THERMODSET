from typing import List
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for the imputation package."""
    columns_with_nan: List[str] = None
    data_path: str = None
    seasonal_period: int = 1440
    knn_neighbors: int = 60
    arima_order: tuple = (1, 1, 1)
    evaluation_path : str = None
    def __post_init__(self):
        if self.columns_with_nan is None:
            self.columns_with_nan = ["inside_temperature", "inside_humidity","GHI","DNI","DHI","outside_temperature","outside_humidity","BP","WS","WD_Avg","WSgust_Max"]
        if self.data_path is None:
            self.data_path = "../../data/processed/final_merged_data.csv"
        if self.evaluation_path is None:
            self.evaluation_path = "/evaluation"