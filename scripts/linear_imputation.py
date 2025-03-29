import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scripts.GEP_Json_Data_To_CSV import calculate_and_save_missing_percentages

# Directory setup
processed_data_dir = "../data/processed/"
plot_dir = os.path.join(processed_data_dir, "quality_plots_after_imputation")
os.makedirs(plot_dir, exist_ok=True)


def forecast_feb_data(year_suffix=None):
    """
    Forecast February 2024 data using February 2023 patterns based on external environmental data
    and time features, while maintaining continuity with January and March 2024.

    Args:
        year_suffix (list, optional): List of year suffixes to process. Defaults to ["_23", "_24"].
    """
    if year_suffix is None:
        year_suffix = ["_23", "_24"]

    processed_data_dir = "../data/processed/"

    # Environmental features to consider for pattern matching
    env_features = [
        'GHI', 'DNI', 'DHI', 'Tair', 'RH',
        'BP', 'WS', 'WD', 'Rain_mm_Tot'
    ]

    # Feature importance weights
    feature_weights = {
        'GHI': 2.0,
        'DNI': 1.5,
        'DHI': 1.5,
        'Tair': 2.0,
        'RH': 1.8,
        'BP': 0.8,
        'WS': 1.5,
        'WD': 0.7,
        'Rain_mm_Tot': 1.7,
        'hour': 3.0,
        'day_of_week': 1.5,
        'is_weekend': 1.0
    }

    for zone in ["Zone_1", "Zone_2"]:
        print(f"\nProcessing {zone} for forecasting February 2024 data...")

        # Load imputed data for both years
        data_2023 = pd.read_csv(os.path.join(processed_data_dir, f"imputed_data_{zone}_23.csv"))
        data_2024 = pd.read_csv(os.path.join(processed_data_dir, f"imputed_data_{zone}_24.csv"))

        # Convert time columns to datetime
        data_2023['time'] = pd.to_datetime(data_2023['time'])
        data_2024['time'] = pd.to_datetime(data_2024['time'])

        # Add time features
        for df in [data_2023, data_2024]:
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day

        # Extract February data for both years
        feb_2023 = data_2023[data_2023['month'] == 2].copy()
        feb_2024 = data_2024[data_2024['month'] == 2].copy()

        # Extract January and March 2024 for trend continuity
        jan_2024 = data_2024[data_2024['month'] == 1].copy()
        mar_2024 = data_2024[data_2024['month'] == 3].copy()

        # Check if February 2024 has missing temperature/humidity
        missing_temp = (feb_2024['temperature'] == -1).sum()
        missing_hum = (feb_2024['humidity'] == -1).sum()

        if missing_temp == 0 and missing_hum == 0:
            print(f"No missing data in February 2024 for {zone}. Skipping...")
            continue

        print(f"Found {missing_temp} missing temperature values and {missing_hum} missing humidity values")

        # Create a function to match patterns and forecast values
        def forecast_missing_values(device_id=None, house=None):
            """Forecast missing values for a specific device and house combination"""

            # Filter data by device and house if provided
            if device_id is not None and house is not None:
                feb23_device = feb_2023[(feb_2023['device_id'] == device_id) & (feb_2023['house'] == house)]
                feb24_device = feb_2024[(feb_2024['device_id'] == device_id) & (feb_2024['house'] == house)]
                jan24_device = jan_2024[(jan_2024['device_id'] == device_id) & (jan_2024['house'] == house)]
                mar24_device = mar_2024[(mar_2024['device_id'] == device_id) & (mar_2024['house'] == house)]
            else:
                feb23_device = feb_2023
                feb24_device = feb_2024
                jan24_device = jan_2024
                mar24_device = mar_2024

            if feb23_device.empty or feb24_device.empty:
                print(f"No data for device {device_id} in house {house}. Skipping...")
                return

            # Calculate statistics for January and March 2024 to establish trends
            if not jan24_device.empty and not mar24_device.empty:
                jan_avg_temp = jan24_device[jan24_device['temperature'] != -1]['temperature'].mean()
                jan_avg_hum = jan24_device[jan24_device['humidity'] != -1]['humidity'].mean()
                mar_avg_temp = mar24_device[mar24_device['temperature'] != -1]['temperature'].mean()
                mar_avg_hum = mar24_device[mar24_device['humidity'] != -1]['humidity'].mean()

                # Calculate adjustment factors for trend continuity
                temp_diff = mar_avg_temp - jan_avg_temp
                hum_diff = mar_avg_hum - jan_avg_hum

                # Days in February (account for leap year)
                if feb24_device['time'].dt.year.iloc[0] % 4 == 0:
                    feb_days = 29
                else:
                    feb_days = 28

                # Create gradient adjustment factors
                feb24_device['temp_adjustment'] = feb24_device['day'].apply(
                    lambda d: (temp_diff * d / feb_days)
                )
                feb24_device['hum_adjustment'] = feb24_device['day'].apply(
                    lambda d: (hum_diff * d / feb_days)
                )
            else:
                # If January or March data is not available, no trend adjustment
                feb24_device['temp_adjustment'] = 0
                feb24_device['hum_adjustment'] = 0

            # Check which environmental features are actually in the data
            available_env_features = [f for f in env_features if
                                      f in feb23_device.columns and f in feb24_device.columns]
            print(f"Available environmental features: {available_env_features}")

            # Include time features
            features_to_use = available_env_features + ['hour', 'day_of_week', 'is_weekend']

            # Check for NaN values in the features
            for df_name, df in [('Feb 2023', feb23_device), ('Feb 2024', feb24_device)]:
                nan_counts = df[features_to_use].isna().sum()
                print(f"NaN values in {df_name} features:")
                for feature, count in nan_counts.items():
                    if count > 0:
                        print(f"  {feature}: {count} NaNs")

            # Handle NaN values in features
            for feature in features_to_use:
                # For 2023 data (reference data)
                if feb23_device[feature].isna().any():
                    # For environmental data, use forward fill, backward fill, and then median imputation
                    feb23_device[feature] = feb23_device[feature].fillna(method='ffill')
                    feb23_device[feature] = feb23_device[feature].fillna(method='bfill')
                    # If still NaN, use median
                    if feb23_device[feature].isna().any():
                        feb23_device[feature] = feb23_device[feature].fillna(feb23_device[feature].median())
                    # If median is NaN (all values are NaN), use 0
                    if feb23_device[feature].isna().any():
                        print(f"Warning: All values for {feature} in Feb 2023 are NaN. Using 0.")
                        feb23_device[feature] = feb23_device[feature].fillna(0)

                # For 2024 data (target data)
                if feb24_device[feature].isna().any():
                    feb24_device[feature] = feb24_device[feature].fillna(method='ffill')
                    feb24_device[feature] = feb24_device[feature].fillna(method='bfill')
                    if feb24_device[feature].isna().any():
                        feb24_device[feature] = feb24_device[feature].fillna(feb24_device[feature].median())
                    if feb24_device[feature].isna().any():
                        print(f"Warning: All values for {feature} in Feb 2024 are NaN. Using 0.")
                        feb24_device[feature] = feb24_device[feature].fillna(0)

            # Check for any remaining NaN values after imputation
            for df_name, df in [('Feb 2023', feb23_device), ('Feb 2024', feb24_device)]:
                if df[features_to_use].isna().any().any():
                    print(f"Warning: Still have NaN values in {df_name} after imputation.")
                    for feature in features_to_use:
                        nan_count = df[feature].isna().sum()
                        if nan_count > 0:
                            print(f"  {feature}: {nan_count} NaNs")
                    # Final fallback - replace any remaining NaNs with 0
                    df[features_to_use] = df[features_to_use].fillna(0)
                    print("All remaining NaN values replaced with 0.")

            # Create feature arrays with weights
            weights = np.array([feature_weights.get(f, 1.0) for f in features_to_use])

            # Extract feature matrices
            X_2023 = feb23_device[features_to_use].values
            X_2024 = feb24_device[features_to_use].values

            # Double-check for NaN values
            if np.isnan(X_2023).any():
                print("Warning: NaN values found in X_2023 despite imputation.")
                X_2023 = np.nan_to_num(X_2023, nan=0.0)

            if np.isnan(X_2024).any():
                print("Warning: NaN values found in X_2024 despite imputation.")
                X_2024 = np.nan_to_num(X_2024, nan=0.0)

            # Normalize features for better distance calculation
            scaler = StandardScaler()
            try:
                X_2023_scaled = scaler.fit_transform(X_2023)
                X_2024_scaled = scaler.transform(X_2024)
            except Exception as e:
                print(f"Error during scaling: {e}")
                # If scaling fails, try a simpler normalization
                print("Falling back to simpler scaling method...")

                # Calculate means and stds, handling zeros in std
                means = np.nanmean(X_2023, axis=0)
                stds = np.nanstd(X_2023, axis=0)
                stds[stds == 0] = 1.0  # Avoid division by zero

                X_2023_scaled = (X_2023 - means) / stds
                X_2024_scaled = (X_2024 - means) / stds

                # Replace NaNs with 0
                X_2023_scaled = np.nan_to_num(X_2023_scaled)
                X_2024_scaled = np.nan_to_num(X_2024_scaled)

            # Apply weights to scaled features
            X_2023_weighted = X_2023_scaled * weights
            X_2024_weighted = X_2024_scaled * weights

            # Create KNN model to find similar patterns
            knn = NearestNeighbors(n_neighbors=min(5, len(X_2023_weighted)), algorithm='ball_tree')

            try:
                knn.fit(X_2023_weighted)
            except Exception as e:
                print(f"Error fitting KNN: {e}")
                print("X_2023_weighted shape:", X_2023_weighted.shape)
                print("X_2023_weighted contains NaN:", np.isnan(X_2023_weighted).any())
                print("X_2023_weighted contains inf:", np.isinf(X_2023_weighted).any())
                # Try again with sanitized data
                X_2023_weighted = np.nan_to_num(X_2023_weighted, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    knn.fit(X_2023_weighted)
                except Exception as e:
                    print(f"Still failed to fit KNN after sanitizing data: {e}")
                    print("Skipping KNN-based forecasting for this device/house.")
                    return

            # Find similar days in 2023 for each day in 2024
            try:
                # Ensure X_2024_weighted is sanitized
                X_2024_weighted = np.nan_to_num(X_2024_weighted, nan=0.0, posinf=0.0, neginf=0.0)
                distances, indices = knn.kneighbors(X_2024_weighted)
            except Exception as e:
                print(f"Error during kNN neighbors search: {e}")
                print("Skipping KNN-based forecasting for this device/house.")
                return

            # Store original indices of feb24_device
            feb24_indices = feb24_device.index.tolist()

            # Process each timestamp in February 2024
            for i, (idx, row) in enumerate(feb24_device.iterrows()):
                # Only process rows with missing temperature or humidity
                if row['temperature'] != -1 and row['humidity'] != -1:
                    continue

                # Get similar patterns
                similar_indices = indices[i]
                similar_rows = feb23_device.iloc[similar_indices]

                # Calculate weighted average based on similarity (inverse distance)
                weights = 1 / (distances[i] + 0.00001)  # Avoid division by zero
                weights = weights / weights.sum()  # Normalize weights

                # Forecast temperature if missing
                if row['temperature'] == -1:
                    forecasted_temp = np.average(similar_rows['temperature'], weights=weights)
                    # Apply trend adjustment
                    forecasted_temp += row['temp_adjustment']
                    # Update the temperature in the original dataframe
                    data_2024.loc[idx, 'temperature'] = forecasted_temp

                # Forecast humidity if missing
                if row['humidity'] == -1:
                    forecasted_hum = np.average(similar_rows['humidity'], weights=weights)
                    # Apply trend adjustment
                    forecasted_hum += row['hum_adjustment']
                    # Ensure humidity is within valid range (0-100%)
                    forecasted_hum = max(0, min(100, forecasted_hum))
                    # Update the humidity in the original dataframe
                    data_2024.loc[idx, 'humidity'] = forecasted_hum

        # Process each device/house combination separately
        device_house_combos = feb_2024[['device_id', 'house']].drop_duplicates()
        for _, combo in device_house_combos.iterrows():
            print(f"Processing device {combo['device_id']} in house {combo['house']}...")
            forecast_missing_values(combo['device_id'], combo['house'])

        # Save the forecasted data
        output_file = os.path.join(processed_data_dir, f"forecasted_data_{zone}_24.csv")
        data_2024.to_csv(output_file, index=False)
        print(f"Forecasted data saved to: {output_file}")

        # Print statistics about the forecasting
        print("\nAfter forecasting:")
        print(f"Number of -1 values in temperature: {(data_2024['temperature'] == -1).sum():,}")
        print(f"Number of -1 values in humidity: {(data_2024['humidity'] == -1).sum():,}")

        # Create a validation plot to check the forecasted data
        try:
            create_validation_plot(data_2024, feb_2023, zone)
        except Exception as e:
            print(f"Error creating validation plots: {e}")

def create_validation_plot(data_2024, feb_2023, zone):
    """
    Create validation plots to visualize the forecasted data against the trend

    Args:
        data_2024 (pd.DataFrame): The complete 2024 dataset with forecasted values
        feb_2023 (pd.DataFrame): February 2023 data used for reference
        zone (str): Zone name for plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    # Extract February 2024 data
    feb_2024 = data_2024[data_2024['time'].dt.month == 2].copy()

    # Get unique device/house combinations
    device_house_combos = feb_2024[['device_id', 'house']].drop_duplicates()

    # Create validation plots
    for _, combo in device_house_combos.iterrows():
        device_id = combo['device_id']
        house = combo['house']

        # Filter data
        feb24_device = feb_2024[(feb_2024['device_id'] == device_id) & (feb_2024['house'] == house)]
        feb23_device = feb_2023[(feb_2023['device_id'] == device_id) & (feb_2023['house'] == house)]

        if feb24_device.empty or feb23_device.empty:
            continue

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot temperature
        ax1.plot(feb23_device['time'], feb23_device['temperature'], 'b-', alpha=0.5, label='Feb 2023')
        ax1.plot(feb24_device['time'], feb24_device['temperature'], 'r-', label='Feb 2024 (Forecasted)')
        ax1.set_ylabel('Temperature')
        ax1.set_title(f'{zone} - Device {device_id} in House {house} - Forecasted February 2024')
        ax1.legend()
        ax1.grid(True)

        # Plot humidity
        ax2.plot(feb23_device['time'], feb23_device['humidity'], 'b-', alpha=0.5, label='Feb 2023')
        ax2.plot(feb24_device['time'], feb24_device['humidity'], 'r-', label='Feb 2024 (Forecasted)')
        ax2.set_ylabel('Humidity')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True)

        # Format x-axis
        date_formatter = DateFormatter('%d-%b')
        ax2.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()

        # Save plot
        plot_dir = "../data/processed/forecast_validation_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"forecast_{zone}_device{device_id}_house{house}.png"), dpi=300,
                    bbox_inches='tight')
        plt.close()


def analyze_and_visualize_gaps(aligned_data, year_suffix=""):
    """
    Analyze recording gaps and create detailed visualizations.
    Blue: humidity = -1
    Red: temperature = -1
    Black: both = -1 or NaN

    Filled dots: all values in 15-min interval are missing
    Empty circles: some values in 15-min interval are missing

    Args:
        aligned_data (pd.DataFrame): The aligned data to analyze
        year_suffix (str): Optional suffix to add to output files (e.g., "_24")

    Returns:
        pd.DataFrame: DataFrame containing gap analysis details
    """
    # Convert to datetime if needed
    plot_dir = os.path.join(processed_data_dir, "quality_plots_after_imputation")
    os.makedirs(plot_dir, exist_ok=True)

    aligned_data['time'] = pd.to_datetime(aligned_data['time'])

    # Create year-month column for grouping
    aligned_data['year_month'] = aligned_data['time'].dt.to_period('M')

    # Initialize gap details list
    gap_details = []

    # Process each zone
    for zone, group in aligned_data.groupby('zone'):
        # Process data in 15-minute intervals
        for interval_start in pd.date_range(start=group['time'].min(),
                                            end=group['time'].max(),
                                            freq='15min'):
            interval_end = interval_start + pd.Timedelta(minutes=15)

            # Get data for this interval
            interval_data = group[
                (group['time'] >= interval_start) &
                (group['time'] < interval_end)
                ]

            if not interval_data.empty:
                # Count total measurements and missing values in the interval
                total_measurements = len(interval_data)
                temp_minus_one_count = (interval_data['temperature'] == -1).sum()
                hum_minus_one_count = (interval_data['humidity'] == -1).sum()
                temp_nan_count = interval_data['temperature'].isna().sum()
                hum_nan_count = interval_data['humidity'].isna().sum()

                # Calculate missing percentages
                temp_missing = temp_minus_one_count + temp_nan_count
                hum_missing = hum_minus_one_count + hum_nan_count

                # Determine gap type and completeness
                if (temp_missing == total_measurements) and (hum_missing == total_measurements):
                    gap_type = 'both'
                    completeness = 'complete'
                elif temp_missing == total_measurements and hum_missing < total_measurements:
                    gap_type = 'temperature'
                    completeness = 'complete'
                elif hum_missing == total_measurements and temp_missing < total_measurements:
                    gap_type = 'humidity'
                    completeness = 'complete'
                elif temp_missing > 0 and hum_missing > 0:
                    gap_type = 'both'
                    completeness = 'partial'
                elif temp_missing > 0:
                    gap_type = 'temperature'
                    completeness = 'partial'
                elif hum_missing > 0:
                    gap_type = 'humidity'
                    completeness = 'partial'
                else:
                    continue  # No gap in this interval

                gap_details.append({
                    'zone': zone,
                    'time': interval_start,
                    'hour': interval_start.hour,
                    'minute': interval_start.minute,
                    'year_month': interval_start.to_period('M'),
                    'gap_type': gap_type,
                    'completeness': completeness,
                    'day': interval_start.day
                })

    gap_df = pd.DataFrame(gap_details)

    # Create visualizations
    print("\nGenerating gap visualizations...")

    # Plot gaps for each zone and month
    for (zone, month), group in gap_df.groupby(['zone', 'year_month']):
        if group.empty:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Define markers for complete and partial gaps
        complete_marker = 'o'  # filled circle
        partial_marker = 'o'  # empty circle

        # Plot different types of gaps
        for gap_type, color, label in [
            ('temperature', 'red', 'Temperature Missing'),
            ('humidity', 'blue', 'Humidity Missing'),
            ('both', 'black', 'Both Missing or NaN')
        ]:
            # Complete gaps (filled markers)
            complete_gaps = group[
                (group['gap_type'] == gap_type) &
                (group['completeness'] == 'complete')
                ]
            if not complete_gaps.empty:
                ax.scatter(
                    complete_gaps['day'],
                    complete_gaps['hour'] + complete_gaps['minute'] / 60,
                    color=color,
                    alpha=0.7,
                    s=20,
                    label=f'{label} (Complete)',
                    marker=complete_marker
                )

            # Partial gaps (empty markers)
            partial_gaps = group[
                (group['gap_type'] == gap_type) &
                (group['completeness'] == 'partial')
                ]
            if not partial_gaps.empty:
                ax.scatter(
                    partial_gaps['day'],
                    partial_gaps['hour'] + partial_gaps['minute'] / 60,
                    facecolors='none',
                    edgecolors=color,
                    alpha=0.7,
                    s=20,
                    label=f'{label} (Partial)',
                    marker=partial_marker
                )

        # Customize the plot
        ax.set_title(f"{zone} - {month.strftime('%B %Y')}", pad=20)
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Hour of Day")

        # Set x-axis ticks to show all days of the month
        ax.set_xticks(range(1, 32))
        ax.set_xlim(0.5, 31.5)

        # Set y-axis to show hours
        ax.set_yticks(range(0, 24, 2))
        ax.set_ylim(-0.5, 23.5)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add detailed explanation
        explanation_text = (
            "Gap Analysis Details:\n"
            "• Time interval: 15 minutes\n"
            "• Filled dots: All values missing\n"
            "• Empty circles: Some values missing\n"
            "• Red: Temperature missing\n"
            "• Blue: Humidity missing\n"
            "• Black: Both missing or NaN"
        )
        ax.text(0.02, 1.02,
                explanation_text,
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

        # Make the plot more compact
        plt.tight_layout()

        # Save plot with optional year suffix
        filename = f"detailed_gaps_{zone.replace(' ', '_')}_{month.strftime('%Y-%m')}_{year_suffix}.png"
        plt.savefig(os.path.join(plot_dir, "detailed_gaps_2024", filename), dpi=300, bbox_inches='tight')
        plt.close()

    return gap_df


def perform_enhanced_imputation(df, interval_minutes: int = 15, window_size: int = 4):
    """
    Perform enhanced linear imputation on temperature and humidity data with expanded window size.
    Handles partial gaps by looking at neighboring intervals for valid values.

    Args:
        df (pd.DataFrame): Input DataFrame with 'time', 'temperature', and 'humidity' columns
        interval_minutes (int): Size of the time interval in minutes
        window_size (int): Number of intervals to look before and after for valid values

    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    # Create a copy of the input DataFrame
    df_imputed = df.copy()

    # Convert time to datetime if not already
    df_imputed['time'] = pd.to_datetime(df_imputed['time'])

    # Process each device and house combination separately
    for (device, house), group in df_imputed.groupby(['device_id', 'house']):
        # Sort by time
        group = group.sort_values('time')

        # Create intervals with expanded window
        interval_starts = pd.date_range(
            start=group['time'].min().floor(f'{interval_minutes}min'),
            end=group['time'].max(),
            freq=f'{interval_minutes}min'
        )

        # Process each interval with expanded window
        for interval_start in interval_starts:
            # Ensure interval_start is a pd.Timestamp
            if not isinstance(interval_start, pd.Timestamp):
                raise TypeError(f"Expected pd.Timestamp, got {type(interval_start)}")

            interval_end = interval_start + pd.Timedelta(minutes=interval_minutes)

            # Define expanded window boundaries
            window_start = interval_start - pd.Timedelta(minutes=interval_minutes * window_size)
            window_end = interval_end + pd.Timedelta(minutes=interval_minutes * window_size)

            # Get data for expanded window
            window_mask = (group['time'] >= window_start) & (group['time'] < window_end)
            window_data = group[window_mask]

            # Get data for current interval
            interval_mask = (group['time'] >= interval_start) & (group['time'] < interval_end)
            interval_data = group[interval_mask]

            if len(interval_data) == 0:
                continue

            # Process temperature
            temp_values = interval_data['temperature'].values
            temp_missing = (temp_values == -1)

            if any(temp_missing):
                # Get valid temperatures from expanded window
                window_temps = window_data['temperature'].values
                window_temps[window_temps == -1] = np.nan

                # Create time indices for interpolation
                window_minutes = (window_data['time'] - window_start).dt.total_seconds() / 60
                temp_series = pd.Series(window_temps, index=window_minutes)

                # Interpolate using the expanded window
                temp_series = temp_series.interpolate(method='linear', limit_direction='both')

                # Extract interpolated values for the current interval
                interval_minutes_values = (interval_data['time'] - window_start).dt.total_seconds() / 60
                # Convert to DataFrame and drop duplicates by averaging duplicate indices
                temp_df = temp_series.to_frame().groupby(temp_series.index).mean()

                # Reindex after removing duplicates
                interpolated_temps = temp_df.reindex(interval_minutes_values).interpolate(method='linear')

                # Update values
                df_imputed.loc[interval_data.index, 'temperature'] = interpolated_temps.values

            # Process humidity (similar logic)
            hum_values = interval_data['humidity'].values
            hum_missing = (hum_values == -1)

            if any(hum_missing):
                # Get valid humidity from expanded window
                window_hums = window_data['humidity'].values
                window_hums[window_hums == -1] = np.nan

                # Create time indices for interpolation
                window_minutes = (window_data['time'] - window_start).dt.total_seconds() / 60
                hum_series = pd.Series(window_hums, index=window_minutes)

                # Interpolate using the expanded window
                hum_series = hum_series.interpolate(method='linear', limit_direction='both')

                # Extract interpolated values for the current interval
                interval_minutes_values = (interval_data['time'] - window_start).dt.total_seconds() / 60
                # Convert to DataFrame and drop duplicates by averaging duplicate indices
                temp_df = hum_series.to_frame().groupby(hum_series.index).mean()

                # Reindex after removing duplicates
                interpolated_hums = temp_df.reindex(interval_minutes_values).interpolate(method='linear')

                # Update values
                df_imputed.loc[interval_data.index, 'humidity'] = interpolated_hums.values

    df_imputed[['temperature', 'humidity']] = df_imputed[['temperature', 'humidity']].fillna(-1)

    return df_imputed


def main():

    year_sufix = ["_23"]

    for year in year_sufix:
        print(f"\nProcessing data for year {year}...")
        # Process Zone 1 and Zone 2 data
        for zone in ["Zone_1", "Zone_2"]:
            print(f"\nProcessing {zone.replace('_', ' ')}...")

            # Load the final merged data
            input_file = os.path.join(processed_data_dir, f"final_merged_data_{zone}{year}.csv")
            df = pd.read_csv(input_file)

            print(f"Original data shape: {df.shape}")
            print("\nBefore imputation:")
            print(f"Number of -1 values in temperature: {(df['temperature'] == -1).sum():,}")
            print(f"Number of -1 values in humidity: {(df['humidity'] == -1).sum():,}")

            # Perform linear imputation
            df_imputed = perform_enhanced_imputation(df)

            print("\nAfter imputation:")
            print(f"Number of -1 values in temperature: {(df_imputed['temperature'] == -1).sum():,}")
            print(f"Number of -1 values in humidity: {(df_imputed['humidity'] == -1).sum():,}")

            # Save imputed data
            output_file = os.path.join(processed_data_dir, f"imputed_data_{zone}{year}.csv")
            df_imputed.to_csv(output_file, index=False)
            print(f"\nImputed data saved to: {output_file}")

            # Generate gap visualization for imputed data
            print("\nGenerating gap visualization for imputed data...")
            analyze_and_visualize_gaps(df_imputed,year)
            print(f"Gap visualization completed for {zone.replace('_', ' ')}")

    for zone in ['Zone_1','Zone_2']:
        imputed_data_file = os.path.join(processed_data_dir, f"imputed_data_{zone}_23.csv")
        df = pd.read_csv(imputed_data_file)
        calculate_and_save_missing_percentages(
            df,
            os.path.join(processed_data_dir + "/missingComparison2023", "missing_percentages_"+zone+"_after_linearImputation_2023.json")
        )
if __name__ == "__main__":
    main()