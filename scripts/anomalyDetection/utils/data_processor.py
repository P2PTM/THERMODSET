import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    """
    Preprocess data with null value handling and feature engineering

    Parameters:
        data (pd.DataFrame): Raw input data

    Returns:
        tuple: Processed data and StandardScaler object
    """
    data['time'] = pd.to_datetime(data['time'])

    processed_data = data.copy()

    # Drop house column if exists
    if 'house' in processed_data.columns:
        processed_data = processed_data.drop('house', axis=1)

    # Time features
    processed_data['hour'] = processed_data['time'].dt.hour
    processed_data['day_of_week'] = processed_data['time'].dt.dayofweek
    processed_data['month'] = processed_data['time'].dt.month
    processed_data['day'] = processed_data['time'].dt.day
    processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)

    # Handle missing values if we use data other than the imputed data
    numerical_cols = ['outside_temperature', 'outside_humidity', 'BP', 'WS', 'WD_Avg',
                      'WSgust_Max', 'Rain_mm_Tot', 'hour', 'day_of_week', 'month', 'day']

    # Interpolate and fill numerical columns
    for col in numerical_cols:
        if col in processed_data.columns:
            processed_data[col] = (processed_data[col]
                                   .interpolate(method='linear')
                                   .fillna(method='ffill')
                                   .fillna(method='bfill'))

    # Fill categorical columns with mode
    categorical_cols = processed_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'time':
            processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])

    # One-hot encode categorical features
    processed_data = pd.get_dummies(processed_data, columns=['zone', 'device_id'])

    # Scale numerical features
    scaler = StandardScaler()
    processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])

    # Verify no nulls remain
    assert processed_data.isnull().sum().sum() == 0, "Null values remain after preprocessing"

    return processed_data, scaler