import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel
from scripts.anomalyDetection.config import config
from scripts.anomalyDetection.utils.logger import logger


class XGBoostDetector(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")

    def train(self, X_train, y_train):
        logger.info("Training XGBoost model...")
        base_model = xgb.XGBClassifier(
            random_state=config.TRAINING.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=config.MODELS.xgboost_params,
            scoring='f1',
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
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