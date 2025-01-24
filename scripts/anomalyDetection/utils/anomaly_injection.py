import numpy as np
import pandas as pd


def inject_and_label_anomalies(data: pd.DataFrame, target_anomaly_rate: float = 0.07):
    np.random.seed(42)
    modified_data = data.copy()

    # Identify numerical columns
    numerical_columns = modified_data.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = [col for col in numerical_columns if col not in ['time']]

    # Inject noise to all numerical columns
    for col in numerical_columns:
        noise = np.random.normal(0, modified_data[col].std() * 0.1, size=len(modified_data))
        modified_data[col] += noise

    # Label anomalies and inject extreme values
    num_anomalies = int(len(modified_data) * target_anomaly_rate)
    anomaly_indices = np.random.choice(modified_data.index, size=num_anomalies, replace=False)

    labels = pd.DataFrame({'anomaly': 0}, index=modified_data.index)
    labels.loc[anomaly_indices, 'anomaly'] = 1

    # Inject illogical values only for inside temperature and inside humidity
    for idx in anomaly_indices:
        modified_data.at[idx, 'inside_temperature'] += np.random.uniform(50, 100)
        modified_data.at[idx, 'inside_humidity'] += np.random.uniform(20, 50)

    return modified_data, labels