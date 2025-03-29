
import os
import argparse
import time
from models_trainer import train_models
from impute_missing import predict_and_impute


def run_pipeline(training_data_path, impute_data_path, output_dir="models", imputed_output_path=None):
    """
    Run the full imputation pipeline: train models and impute missing values.

    Parameters:
    -----------
    training_data_path : str
        Path to the training data CSV file
    impute_data_path : str
        Path to the data CSV file with missing values to impute
    output_dir : str, default="models"
        Directory to save the trained models
    imputed_output_path : str, optional
        Path to save the imputed data CSV file
    """
    start_time = time.time()

    print("=" * 50)
    print("WEATHER DATA IMPUTATION PIPELINE")
    print("=" * 50)

    # Step 1: Train models
    print("\nSTEP 1: Training XGBoost models")
    print("-" * 50)
    metrics = train_models(training_data_path, output_dir)

    # Step 2: Impute missing values
    print("\nSTEP 2: Imputing missing values")
    print("-" * 50)

    if imputed_output_path is None:
        # Generate default output path
        base_dir = os.path.dirname(impute_data_path)
        base_name = os.path.basename(impute_data_path)
        imputed_output_path = os.path.join(base_dir, f"imputed_{base_name}")

    imputed_df = predict_and_impute(impute_data_path, output_dir, imputed_output_path)

    imputed_df.to_csv(imputed_output_path)
    # Summary
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED")
    print("=" * 50)
    print(f"Runtime: {duration:.2f} seconds")
    print(f"Temperature model R²: {metrics['temperature']['r2']:.4f}")
    print(f"Humidity model R²: {metrics['humidity']['r2']:.4f}")
    print(f"Imputed data saved to: {imputed_output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the full imputation pipeline')

    parser.add_argument(
        '--train-data',
        type=str,
        default='../../../data/processed/custom_imputation/forecasted/imputed_forecasted_data_2024_zone_1.csv',
        help='Path to the training data CSV file'
    )

    parser.add_argument(
        '--impute-data',
        type=str,
        default='../../../data/processed/custom_imputation/2023/spline/spline_imputed_2023_zone_2.csv',
        help='Path to the data CSV file with missing values to impute'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save the trained models (default: models)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../../../data/processed/custom_imputation/2023/ml_based/imputed_ml_based_2023_Zone_2.csv',
        help='Path to save the imputed data CSV file'
    )

    args = parser.parse_args()

    run_pipeline(
        training_data_path=args.train_data,
        impute_data_path=args.impute_data,
        output_dir=args.model_dir,
        imputed_output_path=args.output
    )