import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


class MlBasedImputer:
    def __init__(self, temp_model_path='temp_model.json', humidity_model_path='humidity_model.json'):
        """
        Initialize the MlBasedImputer with model paths.

        Parameters:
        -----------
        temp_model_path : str, default='temp_model.json'
            Path to save/load the temperature XGBoost model
        humidity_model_path : str, default='humidity_model.json'
            Path to save/load the humidity XGBoost model
        """
        self.temp_model_path = temp_model_path
        self.humidity_model_path = humidity_model_path
        self.temp_model = None
        self.humidity_model = None
        self.feature_columns = None
        self.metrics = None

    def preprocess_data(self, df):
        """
        Preprocess the data for model training and prediction.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with weather data

        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        """
        # Create a copy to avoid modifying the original dataframe
        data = df.copy()
        data = data.replace(-1, pd.NA)

        # Convert time to datetime if it's not already
        if 'time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['time']):
            data['time'] = pd.to_datetime(data['time'])

        # Extract time-based features if not already present
        if 'hour' not in data.columns and 'time' in data.columns:
            data['hour'] = data['time'].dt.hour
        if 'day_of_week' not in data.columns and 'time' in data.columns:
            data['day_of_week'] = data['time'].dt.dayofweek
        if 'is_weekend' not in data.columns and 'day_of_week' in data.columns:
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        if 'month' not in data.columns and 'time' in data.columns:
            data['month'] = data['time'].dt.month
        if 'day' not in data.columns and 'time' in data.columns:
            data['day'] = data['time'].dt.day

        # Handle categorical variables
        for cat_col in ['zone', 'device_id', 'house']:
            if cat_col in data.columns and not pd.api.types.is_numeric_dtype(data[cat_col]):
                data[cat_col] = data[cat_col].astype('category').cat.codes

        # drop column year_month as it is not useful (it is used for plotting the data)
        if 'year_month' in data.columns:
            data = data.drop('year_month', axis=1)

        if 'WSgust_Max' in data.columns:
            data = data.drop('WSgust_Max', axis=1)
        if 'WD_Std' in data.columns:
            data = data.drop('WD_Std', axis=1)

        # handle NaN values in columns other than temperature and humidity by filling them with linear interpolation
        for col in data.columns:
            if col not in ['temperature', 'humidity'] and data[col].isna().any():
                data[col] = data[col].interpolate(method='linear')

        return data

    def prepare_features(self, df, target=None):
        """
        Prepare features for model training or prediction.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target : str, optional
            Target column name for filtering

        Returns:
        --------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series, optional
            Target vector, only returned if target is provided
        """
        # Define feature columns (excluding target variables and time)
        exclude_cols = ['time', 'temperature', 'humidity']
        if target:
            exclude_cols.remove(target)

        # Use available columns in the dataframe
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # During training, save the feature columns
        if not self.feature_columns and target:
            self.feature_columns = feature_cols

        X = df[feature_cols]

        if target:
            y = df[target]
            return X, y
        else:
            return X

    def train_models(self, df, test_size=0.2, random_state=42):
        """
        Train XGBoost models for temperature and humidity prediction.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with weather data
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random state for reproducibility

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics for both models
        """
        # Preprocess data
        data = self.preprocess_data(df)

        # Check if target columns exist
        if 'temperature' not in data.columns or 'humidity' not in data.columns:
            raise ValueError("Data must contain 'temperature' and 'humidity' columns")

        # Drop rows with NaN values in target columns before training
        data_temp = data.dropna(subset=['temperature'])
        data_hum = data.dropna(subset=['humidity'])

        # Extract features and targets
        all_features = [col for col in data.columns if col not in ['time']]

        # TEMPERATURE MODEL: exclude temperature (target) and humidity (to prevent leakage)
        temp_features = [col for col in all_features if col != 'temperature' and col != 'humidity']
        X_temp = data_temp[temp_features]
        y_temp = data_temp['temperature']

        # HUMIDITY MODEL: exclude humidity (target) and temperature (to prevent leakage)
        hum_features = [col for col in all_features if col != 'humidity' and col != 'temperature']
        X_hum = data_hum[hum_features]
        y_hum = data_hum['humidity']

        # Save distinct feature sets for each model
        joblib.dump(temp_features, 'temp_features.joblib')
        joblib.dump(hum_features, 'hum_features.joblib')

        self.feature_columns = [col for col in all_features if col != 'temperature' and col != 'humidity']

        X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=random_state
        )

        X_hum_train, X_hum_test, y_hum_train, y_hum_test = train_test_split(
            X_hum, y_hum, test_size=test_size, random_state=random_state
        )

        # Train temperature model
        self.temp_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        self.temp_model.fit(X_temp_train, y_temp_train)

        # Train humidity model
        self.humidity_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        self.humidity_model.fit(X_hum_train, y_hum_train)

        # Evaluate models
        temp_preds = self.temp_model.predict(X_temp_test)
        temp_rmse = np.sqrt(mean_squared_error(y_temp_test, temp_preds))
        temp_r2 = r2_score(y_temp_test, temp_preds)

        hum_preds = self.humidity_model.predict(X_hum_test)
        hum_rmse = np.sqrt(mean_squared_error(y_hum_test, hum_preds))
        hum_r2 = r2_score(y_hum_test, hum_preds)

        # Store metrics
        self.metrics = {
            'temperature': {
                'rmse': temp_rmse,
                'r2': temp_r2
            },
            'humidity': {
                'rmse': hum_rmse,
                'r2': hum_r2
            }
        }

        # Save models and metrics
        self.save_models()

        return self.metrics

    def save_models(self):
        """
        Save trained models and other necessary data to files.
        """
        if self.temp_model is not None:
            self.temp_model.save_model(self.temp_model_path)
        if self.humidity_model is not None:
            self.humidity_model.save_model(self.humidity_model_path)

        # Save feature columns
        if self.feature_columns:
            joblib.dump(self.feature_columns, 'feature_columns.joblib')

        # Save metrics
        if self.metrics:
            joblib.dump(self.metrics, 'model_metrics.joblib')

    def load_models(self):
        """
        Load trained models from files.

        Returns:
        --------
        bool
            True if models were loaded successfully, False otherwise
        """
        try:
            # Load feature columns first to ensure we have them
            if os.path.exists('feature_columns.joblib'):
                self.feature_columns = joblib.load('feature_columns.joblib')
            else:
                print("Warning: Feature columns file not found. This may cause issues during prediction.")

            self.temp_model = xgb.XGBRegressor()
            self.temp_model.load_model(self.temp_model_path)

            self.humidity_model = xgb.XGBRegressor()
            self.humidity_model.load_model(self.humidity_model_path)

            # Load metrics
            if os.path.exists('model_metrics.joblib'):
                self.metrics = joblib.load('model_metrics.joblib')

            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False


    def predict(self, df):
        """
        Make predictions using trained models.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with features

        Returns:
        --------
        pandas.DataFrame
            DataFrame with predicted temperature and humidity
        """
        if self.temp_model is None or self.humidity_model is None:
            if not self.load_models():
                raise ValueError("Models not trained or loaded")

        # Preprocess data
        data = self.preprocess_data(df)

        # Prepare features
        X = self.prepare_features(data)

        # Make predictions
        temp_pred = self.temp_model.predict(X)
        hum_pred = self.humidity_model.predict(X)

        # Create output dataframe
        result = pd.DataFrame({
            'predicted_temperature': temp_pred,
            'predicted_humidity': hum_pred
        })

        # Add original index
        result.index = data.index

        return result

    def impute_missing_values(self, df):
        """
        Impute missing temperature and humidity values using separate feature sets.
        """
        data = df.copy()

        # Check if there are missing values
        missing_temp = data['temperature'].isna()
        missing_hum = data['humidity'].isna()

        # Load models and feature sets
        if self.temp_model is None or self.humidity_model is None:
            self.load_models()

        # Load separate feature lists
        temp_features = joblib.load('temp_features.joblib')
        hum_features = joblib.load('hum_features.joblib')

        # Preprocess data
        preprocessed = self.preprocess_data(data)

        # For temperature prediction (excludes both temperature and humidity)
        if missing_temp.any():
            # For rows with missing temperature but available humidity
            X_missing_temp = preprocessed.loc[missing_temp, temp_features]
            data.loc[missing_temp, 'temperature'] = self.temp_model.predict(X_missing_temp)

        # After imputing temperature, update the preprocessed data
        preprocessed = self.preprocess_data(data)

        # For humidity prediction (excludes both humidity and temperature)
        if missing_hum.any():
            X_missing_hum = preprocessed.loc[missing_hum, hum_features]
            data.loc[missing_hum, 'humidity'] = self.humidity_model.predict(X_missing_hum)

        return data


    def plot_feature_importance(self, figsize=(12, 10), save_path='feature_importance.png'):
        """
        Plot feature importance for both models.

        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size for the plot
        save_path : str, default='feature_importance.png'
            Path to save the generated plot

        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the plots
        """
        if self.temp_model is None or self.humidity_model is None:
            if not self.load_models():
                raise ValueError("Models not trained or loaded")

        if self.feature_columns is None:
            raise ValueError("Feature columns not available")

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot temperature feature importance
        temp_importance = self.temp_model.feature_importances_
        temp_indices = np.argsort(temp_importance)[::-1]

        sns.barplot(
            x=temp_importance[temp_indices],
            y=[self.feature_columns[i] for i in temp_indices],
            ax=ax1
        )
        ax1.set_title('Temperature Model Feature Importance')
        ax1.set_xlabel('Importance')
        ax1.set_ylabel('Feature')

        # Plot humidity feature importance
        hum_importance = self.humidity_model.feature_importances_
        hum_indices = np.argsort(hum_importance)[::-1]

        sns.barplot(
            x=hum_importance[hum_indices],
            y=[self.feature_columns[i] for i in hum_indices],
            ax=ax2
        )
        ax2.set_title('Humidity Model Feature Importance')
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Feature')

        plt.tight_layout()
        plt.savefig(save_path)

        return fig

    def get_metrics(self):
        """
        Get model evaluation metrics.

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics for both models
        """
        if self.metrics is None:
            if os.path.exists('model_metrics.joblib'):
                self.metrics = joblib.load('model_metrics.joblib')
            else:
                return {"error": "No metrics available. Train models first."}

        return self.metrics