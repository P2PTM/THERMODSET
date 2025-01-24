from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import numpy as np

from scripts.anomalyDetection.utils.logger import logger


class BaseModel(ABC):
    """Base class for all anomaly detection models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_params = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaseModel':
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        pass

    def save_model(self, path: Path) -> None:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load model from disk"""
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
