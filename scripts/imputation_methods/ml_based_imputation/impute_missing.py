
import pandas as pd
import argparse
import os
import sys
from temp_hum_predictor import MlBasedImputer


def predict_and_impute(data_path, model_dir='models', output_path=None):
    """
    Predict and impute missing temperature and humidity values in a dataset.

    Parameters:
    -----------
    data_path : str
        Path to the data CSV file with missing values
    model_dir : str, default='models'
        Directory containing the trained models
    output_path : str, optional
        Path to save the imputed data CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values
    """
    # Load data
    try:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        original_shape = df.shape
        print(f"Loaded {original_shape[0]} rows and {original_shape[1]} columns.")

        # replace -1 values with NaN
        df = df.replace(-1, pd.NA)

        # Count missing values
        missing_temp = df['temperature'].isna().sum()
        missing_hum = df['humidity'].isna().sum()
        print(f"Found {missing_temp} missing temperature values and {missing_hum} missing humidity values.")

        if missing_temp == 0 and missing_hum == 0:
            print("No missing values to impute.")
            return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

    # Initialize weather predictor
    temp_model_path = os.path.join(model_dir, 'temp_model.json')
    humidity_model_path = os.path.join(model_dir, 'humidity_model.json')

    predictor = MlBasedImputer(
        temp_model_path=temp_model_path,
        humidity_model_path=humidity_model_path
    )

    # Load models
    print("Loading trained models...")
    if not predictor.load_models():
        print(f"Error: Could not load models from {model_dir}")
        sys.exit(1)

    # Get model metrics
    metrics = predictor.get_metrics()
    print("\nModel Metrics:")
    print("-" * 30)
    print(f"Temperature Model - RMSE: {metrics['temperature']['rmse']:.4f}, R²: {metrics['temperature']['r2']:.4f}")
    print(f"Humidity Model - RMSE: {metrics['humidity']['rmse']:.4f}, R²: {metrics['humidity']['r2']:.4f}")

    # Impute missing values
    print("\nImputing missing values...")
    imputed_df = predictor.impute_missing_values(df)

    # Verify imputation
    missing_temp_after = imputed_df['temperature'].isna().sum()
    missing_hum_after = imputed_df['humidity'].isna().sum()
    print(
        f"After imputation: {missing_temp_after} missing temperature values and {missing_hum_after} missing humidity values.")

    # Save imputed data if output path is provided
    if output_path:
        print(f"Saving imputed data to {output_path}...")
        imputed_df.to_csv(output_path, index=False)
        print("Done.")

    return imputed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict and impute missing temperature and humidity values')

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the data CSV file with missing values'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing the trained models (default: models)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='Path to save the imputed data CSV file'
    )

    args = parser.parse_args()

    predict_and_impute(
        data_path=args.data,
        model_dir=args.model_dir,
        output_path=args.output
    )