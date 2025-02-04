import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from scripts.anomalyDetection.config import config
from scripts.anomalyDetection.utils.logger import logger
from scripts.anomalyDetection.models.base_model import BaseModel


class LOFDetector(BaseModel):
    def __init__(self):
        super().__init__("LOF")
        self.model = None
        self.best_params = None

    def train(self, X_train):
        logger.info("Training Local Outlier Factor model...")
        best_score = float('inf')
        best_params = {}

        for n_neighbors in config.MODELS.lof_params['n_neighbors']:
            for contamination in config.MODELS.lof_params['contamination']:
                for metric in config.MODELS.lof_params['metric']:
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination,
                        metric=metric
                    )
                    lof.fit(X_train)

                    # Simple scoring based on negative outlier factor
                    score = -lof.negative_outlier_factor_.mean()

                    if score < best_score:
                        best_score = score
                        best_params = {
                            'n_neighbors': n_neighbors,
                            'contamination': contamination,
                            'metric': metric
                        }

        self.model = LocalOutlierFactor(
            n_neighbors=best_params['n_neighbors'],
            contamination=best_params['contamination'],
            metric=best_params['metric'],
            novelty=True
        )
        self.model.fit(X_train)
        self.best_params = best_params

        logger.info(f"Best parameters: {self.best_params}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return self.model.predict(X)

    def score_samples(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return self.model.score_samples(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        # Compute the negative outlier factor for the input data
        negative_outlier_factor = self.model.score_samples(X)

        # Normalize the negative outlier factor to [0, 1] for probability interpretation
        # Outliers will have a probability closer to 1
        proba = (negative_outlier_factor - negative_outlier_factor.min()) / (
                negative_outlier_factor.max() - negative_outlier_factor.min()
        )
        proba = 1 - proba  # Flip so outliers have higher probabilities

        return np.vstack([1 - proba, proba]).T

