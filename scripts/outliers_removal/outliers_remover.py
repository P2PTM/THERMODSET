import argparse
import logging
import os

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Main pipeline
def outlier_detection_pipeline(df_2023_path, output_path):
    # Load 2023 data
    df_2023 = pd.read_csv(df_2023_path)

    # Convert time to datetime if it exists
    if 'time' in df_2023.columns:
        df_2023['time'] = pd.to_datetime(df_2023['time'])
        # Extract time-based features
        df_2023['month'] = df_2023['time'].dt.month
        df_2023['day'] = df_2023['time'].dt.day
        df_2023['hour'] = df_2023['time'].dt.hour
        df_2023['day_of_week'] = df_2023['time'].dt.dayofweek
        df_2023['is_weekend'] = df_2023['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    else:
        print("Error: 'time' column not found in the dataset.")
        return None

    # Define target features for outlier detection
    target_features = ['temperature', 'humidity']

    feature_configs = init_feature_config(df_2023, target_features)

    # Initialize empty DataFrame to collect cleaned data from all months
    df_clean_all = pd.DataFrame()

    # Process each month separately
    for month in range(1, 13):
        print(f"\n{'=' * 50}")
        print(f"Processing Month {month}")
        print(f"{'=' * 50}")

        # Get data for this month
        df_month = df_2023[df_2023['month'] == month].copy()

        # Skip if no data for this month
        if len(df_month) == 0:
            print(f"No data available for month {month}")
            continue

        print(f"Found {len(df_month)} records for month {month}")

        # Process each feature separately
        for feature in target_features:
            if feature not in df_month.columns:
                continue

            print(f"\nProcessing feature: {feature} for month {month}")

            # Create synthetic dataset with injected noise and labeled outliers for this feature and month
            print(f"Creating synthetic dataset with {feature} outliers")

            df_noisy_feature, features_to_use = create_noise_outliers(df_month, feature, feature_configs)

            # Train model specific to this feature and month
            print(f"Training {feature} outlier detection model for month {month}")
            try:
                model, scaler = train_monthly_model(df_noisy_feature, feature, features_to_use, month)

                impute_outliers(df_month, feature, features_to_use, model, month, scaler)


            except Exception as e:
                print(f"Error processing {feature} for month {month}: {e}")

        # After processing all features for this month, add to the final dataset
        df_clean_all = pd.concat([df_clean_all, df_month])

    # Sort by time
    if len(df_clean_all) > 0:
        df_clean_all = df_clean_all.sort_values('time')

        # Calculate total outliers across all features
        outlier_summary = {}
        for feature in target_features:
            if f'is_{feature[:4]}_outlier' in df_clean_all.columns:
                outlier_count = df_clean_all[f'is_{feature[:4]}_outlier'].sum()
                outlier_summary[feature] = outlier_count
                print(f"Total {feature} outliers identified: {outlier_count}")

        # Save the combined cleaned data
        print(f"\nSaving cleaned data to {output_path}")
        df_clean_all.to_csv(output_path, index=False)
        return df_clean_all
    else:
        print("No data processed successfully")
        return None


def init_feature_config(df_2023, target_features):
    # Create rolling statistics features for temperature and humidity
    for feature in target_features:
        if feature in df_2023.columns:
            # Calculate rolling statistics
            df_2023[f'{feature[:4]}_rolling_mean'] = df_2023[feature].rolling(window=24, min_periods=1).mean()
            df_2023[f'{feature[:4]}_rolling_std'] = df_2023[feature].rolling(window=24, min_periods=1).std().fillna(0)
            # Calculate difference from previous hour
            df_2023[f'{feature[:4]}_diff'] = df_2023[feature].diff().fillna(0)
            # Calculate z-score
            mean = df_2023[feature].mean()
            std = df_2023[feature].std()
            df_2023[f'{feature[:4]}_z_score'] = (df_2023[feature] - mean) / (std if std > 0 else 1)

            # Initialize feature-specific outlier flags
            df_2023[f'is_{feature[:4]}_outlier'] = 0
    # Define base features for all models
    base_features = ['day', 'day_of_week', 'hour', 'is_weekend']
    # Store feature configurations for each target feature
    feature_configs = {}
    for feature in target_features:
        if feature in df_2023.columns:
            specific_features = base_features.copy()
            specific_features.append(feature)
            specific_features.extend([
                f'{feature[:4]}_rolling_mean',
                f'{feature[:4]}_rolling_std',
                f'{feature[:4]}_diff',
                f'{feature[:4]}_z_score'
            ])
            feature_configs[feature] = specific_features
    # Store features list for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(feature_configs, 'models/feature_configs.joblib')
    return feature_configs


def create_noise_outliers(df_month, feature, feature_configs):
    # Create a copy for this feature's outlier detection
    df_noisy_feature = df_month.copy()
    # Use more conservative percentile-based outlier detection for this feature
    lower_percentile = df_noisy_feature[feature].quantile(0.09)
    upper_percentile = df_noisy_feature[feature].quantile(0.91)
    # Identify values outside these percentiles
    outlier_indices = df_noisy_feature[
        (df_noisy_feature[feature] < lower_percentile) |
        (df_noisy_feature[feature] > upper_percentile)
        ].index
    # Mark these as outliers for this specific feature
    df_noisy_feature[f'is_{feature[:4]}_outlier'] = 0
    df_noisy_feature.loc[outlier_indices, f'is_{feature[:4]}_outlier'] = 1
    # More conservative rapid change detection
    changes = df_noisy_feature[feature].diff().abs()
    change_threshold = changes.quantile(0.99)
    outlier_indices = df_noisy_feature[changes > change_threshold].index
    df_noisy_feature.loc[outlier_indices, f'is_{feature[:4]}_outlier'] = 1
    # Use Isolation Forest with reduced contamination
    features_to_use = feature_configs[feature]
    X = df_noisy_feature[features_to_use].fillna(df_noisy_feature[features_to_use].median())
    iso_forest = IsolationForest(contamination=0.001, random_state=42)
    outlier_preds = iso_forest.fit_predict(X)
    outlier_indices = df_noisy_feature.index[outlier_preds == -1]
    df_noisy_feature.loc[outlier_indices, f'is_{feature[:4]}_outlier'] = 1
    # Log outlier statistics
    outlier_count = df_noisy_feature[f'is_{feature[:4]}_outlier'].sum()
    total_count = len(df_noisy_feature)
    print(f"Created dataset with {outlier_count} {feature} outliers out of {total_count} records "
          f"({outlier_count / total_count * 100:.2f}%)")
    return df_noisy_feature, features_to_use


def train_monthly_model(df_noisy_feature, feature, features_to_use, month):
    X = df_noisy_feature[features_to_use].fillna(df_noisy_feature[features_to_use].median())
    y = df_noisy_feature[f'is_{feature[:4]}_outlier']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train XGBoost model instead of Random Forest
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    )
    model.fit(X_train_scaled, y_train)
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"{feature} detection metrics:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    # Save feature-specific, month-specific model
    os.makedirs(f'models/monthly_models/{feature}', exist_ok=True)
    joblib.dump(model, f'models/monthly_models/{feature}/outlier_model_month_{month}.joblib')
    joblib.dump(scaler, f'models/monthly_models/{feature}/outlier_scaler_month_{month}.joblib')
    return model, scaler


def impute_may_june_humidity(df_month, feature):
    # More aggressive outlier detection
    Q1 = df_month[feature].quantile(0.1)
    Q3 = df_month[feature].quantile(0.9)
    IQR = Q3 - Q1
    # Wider bounds to capture more potential outliers
    lower_bound = Q1 - 2 * IQR  # Changed from 1.5 to 3
    upper_bound = Q3 + 2 * IQR  # Changed from 1.5 to 3
    # Identify outliers
    outliers = df_month[
        (df_month[feature] < lower_bound) |
        (df_month[feature] > upper_bound)
        ]
    # Impute outliers with median of non-outlier data for that month
    non_outlier_median = df_month[
        (df_month[feature] >= lower_bound) &
        (df_month[feature] <= upper_bound)
        ][feature].median()
    # Replace outliers with the month's non-outlier median
    df_month.loc[outliers.index, feature] = non_outlier_median


def impute_outliers(df_month, feature, features_to_use, model, month, scaler):
    # Apply model to actual data for this month and feature
    print(f"Detecting {feature} outliers for month {month}")
    X_month = df_month[features_to_use].fillna(df_month[features_to_use].median())
    X_month_scaled = scaler.transform(X_month)
    outlier_predictions = model.predict(X_month_scaled)
    df_month[f'is_{feature[:4]}_outlier'] = outlier_predictions
    # Impute outliers for this specific feature
    print(f"Imputing {feature} outliers for month {month}")
    for (day, hour), group in df_month.groupby(['day', 'hour']):
        # Get indices of outliers in this group
        outlier_indices = group[group[f'is_{feature[:4]}_outlier'] == 1].index

        if len(outlier_indices) > 0:
            # Get median of non-outliers in this group
            median_value = group[group[f'is_{feature[:4]}_outlier'] == 0][feature].median()

            # If no non-outliers in this group, use overall median
            if pd.isna(median_value):
                median_value = df_month[df_month[f'is_{feature[:4]}_outlier'] == 0][feature].median()

                # If still no valid median, use the overall median
                if pd.isna(median_value):
                    median_value = df_month[feature].median()


            df_month.loc[outlier_indices, feature] = median_value


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outlier_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Process sensor data to remove outliers.')
    parser.add_argument('--input_dir', type=str,
                        default='../../data/processed',
                        help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='../../data/processed/custom_imputation/2023/anomalies_free',
                        help='Directory to save processed CSV files')
    parser.add_argument('--model_dir', type=str,
                        default='models/outlier_detection',
                        help='Directory to save/load outlier detection models')
    parser.add_argument('--train', action='store_true',
                        help='Train new outlier detection models')
    parser.add_argument('--file', type=str, default='forecasted_data_Zone_1_24.csv',
                        help='Process a specific file instead of all files in the directory')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process single file or all files
    if args.file:
        input_path = os.path.join(args.input_dir, args.file)
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return

        outlier_detection_pipeline(input_path, args.output_dir + '/anomalies_free_2024_1.csv')


if __name__ == "__main__":
    main()