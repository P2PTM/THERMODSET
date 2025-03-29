
import pandas as pd
import argparse
import os
import sys
from temp_hum_predictor import MlBasedImputer


def train_models(training_data_path, output_dir='models', test_size=0.2, random_state=42):
    """
    Train temperature and humidity prediction models using XGBoost.

    Parameters:
    -----------
    training_data_path : str
        Path to the training data CSV file
    output_dir : str, default='models'
        Directory to save the trained models
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for both models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    try:
        print(f"Loading training data from {training_data_path}...")
        df = pd.read_csv(training_data_path)
        print(f"Loaded {len(df)} rows of data.")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        sys.exit(1)

    # Initialize weather predictor
    temp_model_path = os.path.join(output_dir, 'temp_model.json')
    humidity_model_path = os.path.join(output_dir, 'humidity_model.json')

    predictor = MlBasedImputer(
        temp_model_path=temp_model_path,
        humidity_model_path=humidity_model_path
    )

    # Train models
    print("Training models...")
    metrics = predictor.train_models(df, test_size=test_size, random_state=random_state)

    # Print metrics
    print("\nModel Training Results:")
    print("-" * 30)
    print("Temperature Model:")
    print(f"  RMSE: {metrics['temperature']['rmse']:.4f}")
    print(f"  R²: {metrics['temperature']['r2']:.4f}")
    print("\nHumidity Model:")
    print(f"  RMSE: {metrics['humidity']['rmse']:.4f}")
    print(f"  R²: {metrics['humidity']['r2']:.4f}")

    # Generate feature importance plot
    print("\nGenerating feature importance plot...")
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    predictor.plot_feature_importance(save_path=plot_path)
    print(f"Feature importance plot saved to {plot_path}")

    print(f"\nModels saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost models for temperature and humidity prediction')

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the training data CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Directory to save the trained models (default: models)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    train_models(
        training_data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )