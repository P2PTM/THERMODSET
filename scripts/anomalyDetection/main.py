import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from config import config
from utils.anomaly_injection import inject_and_label_anomalies
from utils.data_processor import preprocess_data
from utils.evaluation import evaluate_model, plot_roc_curves
from utils.logger import logger
from models.xgboost_model import XGBoostDetector
from models.naive_bayes_model import NaiveBayesDetector
from models.lof_model import LOFDetector
from models.autoencoder_model import AutoencoderDetector


def generate_synthetic_data(original_data, n_samples=None):
    """
    Generate synthetic data similar to the original dataset

    Args:
        original_data (pd.DataFrame): Original dataset
        n_samples (int, optional): Number of synthetic samples to generate

    Returns:
        pd.DataFrame: Synthetic dataset
    """
    if n_samples is None:
        n_samples = len(original_data)

    # Resample with replacement
    synthetic_data = resample(original_data, replace=True, n_samples=n_samples)

    # Add some noise
    for col in synthetic_data.columns:
        if col in ['time', 'house','zone','device_id']:
            continue
        synthetic_data[col] += np.random.normal(0, synthetic_data[col].std() * 0.1, size=n_samples)

    return synthetic_data


def train_and_detect_anomalies(data_path):
    """
    Execution function for anomaly detection with synthetic data generation

    Args:
        data_path (str): Path to input data CSV file

    Returns:
        dict: Trained models and their performance metrics
    """
    # Load main data
    main_data = pd.read_csv(data_path)

    # Generate synthetic dataset
    synthetic_data = generate_synthetic_data(main_data)

    # Inject and label anomalies in synthetic data
    synthetic_data_with_anomalies, synthetic_labels = inject_and_label_anomalies(synthetic_data)

    # Preprocess synthetic data
    processed_synthetic_data, scaler = preprocess_data(synthetic_data_with_anomalies)

    # Prepare features for synthetic data
    X_synthetic = processed_synthetic_data.drop(['time'], axis=1)
    y_synthetic = synthetic_labels['anomaly']

    # Train-test split for synthetic data
    X_train, X_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic,
        test_size=config.TRAINING.test_size,
        random_state=config.TRAINING.random_state
    )

    # Initialize models
    models = {
        'XGBoost': XGBoostDetector(),
        'Naive Bayes': NaiveBayesDetector(),
        'LOF': LOFDetector(),
        'Autoencoder': AutoencoderDetector()
    }

    predictions_dict = {}
    model_metrics = {}

    # Train and evaluate each model on synthetic data
    for name, model in models.items():
        logger.info(f"Training {name} model on synthetic data...")

        # Model-specific training and prediction
        if name == 'XGBoost':
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        elif name == 'Naive Bayes':
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        elif name == 'LOF':
            model.train(X_train)
            y_pred = model.predict(X_test)
            y_pred = (y_pred == -1).astype(int)
            y_pred_proba = -model.score_samples(X_test)
        elif name == 'Autoencoder':
            model.train(X_train)
            y_pred, y_pred_proba = model.predict(X_test)

        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, name)
        model_metrics[name] = metrics
        predictions_dict[name] = y_pred_proba

    # Plot ROC curves
    plot_roc_curves(y_test, predictions_dict)

    # Select best model based on F1 score
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k][2])
    best_model = models[best_model_name]

    # Preprocess main data
    main_data_processed, _ = preprocess_data(main_data)
    X_main = main_data_processed.drop(['time'], axis=1)

    # Apply best model to main data
    if best_model_name in ['XGBoost', 'Naive Bayes']:
        y_pred_main = best_model.predict(X_main)
        y_pred_proba_main = best_model.predict_proba(X_main)
    elif best_model_name == 'LOF':
        y_pred_main = best_model.predict(X_main)
        y_pred_main = (y_pred_main == -1).astype(int)
        y_pred_proba_main = -best_model.score_samples(X_main)
    elif best_model_name == 'Autoencoder':
        y_pred_main, y_pred_proba_main = best_model.predict(X_main)

    # Log anomalies in main data
    anomaly_indices = np.where(y_pred_main == 1)[0]
    logger.info(f"Found {len(anomaly_indices)} anomalies in main dataset using {best_model_name}")

    return {
        'models': models,
        'metrics': model_metrics,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'anomalies_in_main_data': anomaly_indices
    }


def main():
    """
    Main function to run anomaly detection pipeline
    """
    # Input data path
    input_path = config.DATA.processed_path + '/' +  config.DATA.input_file

    # Run anomaly detection
    results = train_and_detect_anomalies(input_path)

    # Save the best model
    model_save_path = os.path.join(
        config.DATA.anomaly_free_path,
        f"{results['best_model_name']}_model.joblib"
    )
    results['best_model'].save_model(model_save_path)

    # Save anomaly-free data
    main_data = pd.read_csv(input_path)
    anomaly_mask = np.ones(len(main_data), dtype=bool)
    anomaly_mask[results['anomalies_in_main_data']] = False
    anomaly_free_data = main_data[anomaly_mask]

    output_path = os.path.join(
        config.DATA.anomaly_free_path,
        config.DATA.output_file
    )
    anomaly_free_data.to_csv(output_path, index=False)
    logger.info(f"Anomaly-free data saved to {output_path}")

    # Save configuration
    config_save_path = os.path.join(
        config.DATA.anomaly_free_path,
        'config.yaml'
    )
    config.save_config(config_save_path)

    return results


if __name__ == "__main__":
    main()