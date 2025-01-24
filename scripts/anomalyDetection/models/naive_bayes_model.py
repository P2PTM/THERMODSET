from sklearn.naive_bayes import GaussianNB
from scripts.anomalyDetection.config import config
from scripts.anomalyDetection.utils.logger import logger
from scripts.anomalyDetection.models.base_model import BaseModel


class NaiveBayesDetector(BaseModel):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        logger.info("Training Naive Bayes model...")
        base_model = GaussianNB()

        # Grid search for var_smoothing parameter
        best_score = -float('inf')
        best_var_smoothing = config.MODELS.naive_bayes_params['var_smoothing'][0]

        for var_smoothing in config.MODELS.naive_bayes_params['var_smoothing']:
            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)

            if score > best_score:
                best_score = score
                best_var_smoothing = var_smoothing

        self.model = GaussianNB(var_smoothing=best_var_smoothing)
        self.model.fit(X_train, y_train)
        self.best_params = {'var_smoothing': best_var_smoothing}

        logger.info(f"Best parameters: {self.best_params}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return self.model.predict_proba(X)[:, 1]