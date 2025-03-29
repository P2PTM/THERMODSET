import os
import json
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Global constants
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
PLOT_DIR = os.path.join(PROCESSED_DATA_DIR, "quality_plots")


def setup_directories():
    """Setup necessary directories for processing."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, "detailed_gaps"), exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print("Directories setup complete.")


def extract_zip(zip_path=None):
    """Extract the zip file containing raw data."""
    if not zip_path:
        zip_path = os.path.join(RAW_DATA_DIR, "compressed.zip")

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"Extracted {zip_path} to {RAW_DATA_DIR}")
    else:
        print(f"Warning: Zip file not found at {zip_path}, skipping extraction.")


def merge_temperature_humidity(humidity_df, temperature_df):
    """
    Merge temperature and humidity data, ensuring that if a timestamp exists in one dataset
    but not in the other, it's added with a value of -1 for the missing measure.
    """
    # Convert time columns to datetime if not already
    humidity_df['time'] = pd.to_datetime(humidity_df['time'])
    temperature_df['time'] = pd.to_datetime(temperature_df['time'])

    merged_data = []

    # Process each device and house combination separately
    unique_devices = set(humidity_df['device_id'].unique()) | set(temperature_df['device_id'].unique())
    unique_houses = set(humidity_df['house'].unique()) | set(temperature_df['house'].unique())

    for device in unique_devices:
        for house in unique_houses:
            # Get data for current device and house
            hum_data = humidity_df[(humidity_df['device_id'] == device) &
                                   (humidity_df['house'] == house)]
            temp_data = temperature_df[(temperature_df['device_id'] == device) &
                                       (temperature_df['house'] == house)]

            if hum_data.empty and temp_data.empty:
                continue

            # Get all unique timestamps
            all_times = sorted(set(hum_data['time']) | set(temp_data['time']))

            for time in all_times:
                record = {
                    'time': time,
                    'device_id': device,
                    'house': house
                }

                # Get humidity value if exists
                hum_match = hum_data[hum_data['time'] == time]
                record['humidity'] = hum_match['humidity'].iloc[0] if not hum_match.empty else -1

                # Get temperature value if exists
                temp_match = temp_data[temp_data['time'] == time]
                record['temperature'] = temp_match['temperature'].iloc[0] if not temp_match.empty else -1

                merged_data.append(record)

    return pd.DataFrame(merged_data)


def sync_zones_bidirectional(zone1_data, zone2_data):
    """
    Synchronize data between zones by copying valid values from one zone to another
    when they share the same timestamp and one has invalid data.

    Args:
        zone1_data (pd.DataFrame): Data from Zone 1
        zone2_data (pd.DataFrame): Data from Zone 2

    Returns:
        tuple: Updated Zone 1 and Zone 2 DataFrames
    """
    # Ensure time columns are datetime
    zone1_data['time'] = pd.to_datetime(zone1_data['time'])
    zone2_data['time'] = pd.to_datetime(zone2_data['time'])

    # Create copies to avoid modifying original data
    zone1_updated = zone1_data.copy()
    zone2_updated = zone2_data.copy()

    # Get all unique timestamps from both zones
    all_times = sorted(set(zone1_data['time']) | set(zone2_data['time']))

    print("Starting zone synchronization...")
    total_times = len(all_times)

    for i, time in enumerate(all_times):
        if i % 10000 == 0:  # Progress update every 10000 timestamps
            print(f"Processing timestamp {i}/{total_times}")

        # Get rows for this timestamp
        zone1_row = zone1_updated[zone1_updated['time'] == time]
        zone2_row = zone2_updated[zone2_updated['time'] == time]

        if not zone1_row.empty and not zone2_row.empty:
            # Check temperature values
            temp1 = zone1_row['temperature'].iloc[0]
            temp2 = zone2_row['temperature'].iloc[0]

            # If zone1 has valid data and zone2 doesn't
            if (pd.notna(temp1) and temp1 != -1) and (pd.isna(temp2) or temp2 == -1):
                zone2_updated.loc[zone2_updated['time'] == time, 'temperature'] = temp1
            # If zone2 has valid data and zone1 doesn't
            elif (pd.notna(temp2) and temp2 != -1) and (pd.isna(temp1) or temp1 == -1):
                zone1_updated.loc[zone1_updated['time'] == time, 'temperature'] = temp2

            # Check humidity values
            hum1 = zone1_row['humidity'].iloc[0]
            hum2 = zone2_row['humidity'].iloc[0]

            # If zone1 has valid data and zone2 doesn't
            if (pd.notna(hum1) and hum1 != -1) and (pd.isna(hum2) or hum2 == -1):
                zone2_updated.loc[zone2_updated['time'] == time, 'humidity'] = hum1
            # If zone2 has valid data and zone1 doesn't
            elif (pd.notna(hum2) and hum2 != -1) and (pd.isna(hum1) or hum1 == -1):
                zone1_updated.loc[zone1_updated['time'] == time, 'humidity'] = hum2

    print("Zone synchronization completed")

    # Fill remaining null values with -1
    zone1_updated['temperature'] = zone1_updated['temperature'].fillna(-1)
    zone1_updated['humidity'] = zone1_updated['humidity'].fillna(-1)
    zone2_updated['temperature'] = zone2_updated['temperature'].fillna(-1)
    zone2_updated['humidity'] = zone2_updated['humidity'].fillna(-1)

    return zone1_updated, zone2_updated


def fill_data_gaps(data):
    """
    Fill gaps in the data with -1 values at 1-minute intervals.
    """
    filled_data_list = []

    for (zone, device, house), group in data.groupby(['zone', 'device_id', 'house']):
        # Get min and max time for this group
        min_time = group['time'].min()
        max_time = group['time'].max()

        # Create expected 1-minute intervals
        expected_times = pd.date_range(start=min_time, end=max_time, freq='1min')

        # Create a DataFrame with all expected times
        complete_range_df = pd.DataFrame({
            'time': expected_times,
            'zone': zone,
            'device_id': device,
            'house': house
        })

        # Merge with actual data
        merged = pd.merge(
            complete_range_df,
            group,
            on=['time', 'zone', 'device_id', 'house'],
            how='left'
        )

        # Fill missing values with -1
        merged['temperature'] = merged['temperature'].fillna(-1)
        merged['humidity'] = merged['humidity'].fillna(-1)

        # Add to list of filled data
        filled_data_list.append(merged)

    # Combine all filled data
    filled_data = pd.concat(filled_data_list, ignore_index=True)

    # Sort the filled data
    filled_data = filled_data.sort_values(['zone', 'device_id', 'house', 'time'])

    return filled_data


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
        filename = f"detailed_gaps_{zone.replace(' ', '_')}_{month.strftime('%Y-%m')}{year_suffix}.png"
        plt.savefig(os.path.join(PLOT_DIR, "detailed_gaps", filename), dpi=300, bbox_inches='tight')
        plt.close()

    return gap_df


def print_data_statistics(data, zone):
    """Print statistics about the data quality."""
    total_rows = len(data)
    nan_temp = data['temperature'].isna().sum()
    nan_humidity = data['humidity'].isna().sum()
    minus_one_temp = (data['temperature'] == -1).sum()
    minus_one_humidity = (data['humidity'] == -1).sum()

    stats = f"""Data statistics for {zone}:
    Total rows: {total_rows:,}
    NaN values in temperature: {nan_temp:,}
    NaN values in humidity: {nan_humidity:,}
    '-1' values in temperature: {minus_one_temp:,}
    '-1' values in humidity: {minus_one_humidity:,}
    """
    print(stats)
    return stats


def load_zone_data(file_groups):
    """
    Load and process zone data from the specified file groups.

    Args:
        file_groups (dict): Dictionary mapping zones to periods and files

    Returns:
        dict: Dictionary mapping zones to processed DataFrames
    """
    zone_data_dict = {}

    # Process each zone separately
    for zone, periods in file_groups.items():
        print(f"\nProcessing {zone}")
        zone_data = pd.DataFrame()

        # Process each period within the zone
        for period, files in periods.items():
            print(f"Processing period: {period}")

            # Load both datasets
            with open(files["humidity"], "r") as hum_file:
                humidity_df = pd.DataFrame(json.load(hum_file)["data"])
                humidity_df = humidity_df.rename(columns={"measures": "humidity"})

            with open(files["temperature"], "r") as temp_file:
                temperature_df = pd.DataFrame(json.load(temp_file)["data"])
                temperature_df = temperature_df.rename(columns={"measures": "temperature"})

            # Merge temperature and humidity
            merged_df = merge_temperature_humidity(humidity_df, temperature_df)

            # Add zone information
            merged_df["zone"] = zone

            zone_data = pd.concat([zone_data, merged_df])

        # Fill gaps with -1 values
        print(f"\nFilling gaps with -1 values for {zone}...")
        zone_data = fill_data_gaps(zone_data)

        # Store processed zone data
        zone_data_dict[zone] = zone_data

    return zone_data_dict


def process_zone_data(zone_data_dict, outside_data_file, year_suffix=""):
    """
    Process zone data, including synchronization between zones and merging with outside data.

    Args:
        zone_data_dict (dict): Dictionary mapping zones to DataFrames
        outside_data_file (str): Path to the outside weather data file
        year_suffix (str): Optional suffix to add to output files (e.g., "_24")
    """
    # Sync zones bidirectionally if both Zone 1 and Zone 2 are present
    if "Zone 1" in zone_data_dict and "Zone 2" in zone_data_dict:
        print("\nSynchronizing data between Zone 1 and Zone 2...")
        print(
            f"Initial size - Zone 1: {len(zone_data_dict['Zone 1']):,} rows, Zone 2: {len(zone_data_dict['Zone 2']):,} rows")

        zone_data_dict["Zone 1"], zone_data_dict["Zone 2"] = sync_zones_bidirectional(
            zone_data_dict["Zone 1"],
            zone_data_dict["Zone 2"]
        )
        calculate_and_save_missing_percentages(
            zone_data_dict["Zone 1"],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone1_after_sync_"+year_suffix+".json")
        )
        calculate_and_save_missing_percentages(
            zone_data_dict["Zone 2"],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone2_after_sync_"+year_suffix+".json")
        )
        print(
            f"Final size - Zone 1: {len(zone_data_dict['Zone 1']):,} rows, Zone 2: {len(zone_data_dict['Zone 2']):,} rows")

    # Load outside data
    print(f"\nLoading outside weather data from {outside_data_file}")
    outside_data = pd.read_excel(outside_data_file)

    # Check if the data is 2023 or 2024 format
    timestamp_column = "TMSTAMP" if "TMSTAMP" in outside_data.columns else "TIMESTAMP"

    # Process each zone's data
    for zone, zone_data in zone_data_dict.items():
        print(f"\nProcessing aligned data for {zone}")

        # Sort the zone data
        zone_data = zone_data.sort_values(['device_id', 'house', 'time'])

        # Save the intermediate merged data for this zone
        zone_output_file = os.path.join(PROCESSED_DATA_DIR, f"merged_data_{zone.replace(' ', '_')}{year_suffix}.csv")
        zone_data.to_csv(zone_output_file, index=False)
        print(f"Data for {zone} has been successfully merged and saved to {zone_output_file}")

        # Process and align with outside data
        zone_data["time"] = pd.to_datetime(zone_data["time"])
        outside_data[timestamp_column] = pd.to_datetime(outside_data[timestamp_column])

        # Handle 2023 data (15-min intervals need resampling)
        if timestamp_column == "TMSTAMP":
            outside_data_1min = outside_data.set_index(timestamp_column).resample('1T').ffill().reset_index()
            # Perform the merge_asof
            aligned_data = pd.merge_asof(
                zone_data.sort_values("time"),
                outside_data_1min.sort_values(timestamp_column),
                left_on="time",
                right_on=timestamp_column,
                direction="backward",
                tolerance=pd.Timedelta("1min")
            )
        else:
            # 2024 data is already in 1-min intervals, use regular merge
            aligned_data = pd.merge(
                zone_data,
                outside_data,
                left_on="time",
                right_on=timestamp_column,
                how="left"
            )

        # Drop unnecessary columns
        aligned_data = aligned_data.drop(columns=[timestamp_column])

        # Analyze gaps for this zone
        gap_details = analyze_and_visualize_gaps(aligned_data, year_suffix)

        # Generate and save gap summary
        print(f"\nGap Summary Statistics for {zone}:")
        summary = gap_details.groupby(['zone', 'year_month', 'gap_type']).agg(
            total_intervals=('time', 'count'),
        ).reset_index()
        print("\nDetailed Gap Analysis:")
        print(summary)

        # Save summary to CSV
        summary_path = os.path.join(PROCESSED_DATA_DIR,
                                    f"gap_analysis_summary_{zone.replace(' ', '_')}{year_suffix}.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nGap analysis summary for {zone} has been saved to: {summary_path}")

        # Calculate statistics before filling gaps
        print_data_statistics(aligned_data, zone)

        # Save the final filled data
        final_output_file = os.path.join(PROCESSED_DATA_DIR,
                                         f"final_merged_data_{zone.replace(' ', '_')}{year_suffix}.csv")
        aligned_data.to_csv(final_output_file, index=False)
        print(f"The final merged data with filled gaps for {zone} has been saved to {final_output_file}")


def calculate_and_save_missing_percentages(df, output_filename):
    """
    Calculate missing value percentages for a DataFrame and save to a JSON file.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        output_filename (str): Full path for the output JSON file

    Returns:
        dict: Missing value percentages
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Total rows in the DataFrame
    total_rows = len(df)

    # Calculate missing percentages
    missing_percentages = {
        'total_rows': total_rows,
        'temperature_missing_percentage': (
            (df['temperature'] == -1) |
            (df['temperature'].isna())
        ).sum() / total_rows * 100,
        'humidity_missing_percentage': (
            (df['humidity'] == -1) |
            (df['humidity'].isna())
        ).sum() / total_rows * 100
    }

    # Save to JSON file
    with open(output_filename, 'w') as f:
        json.dump(missing_percentages, f, indent=4)

    print(f"\nMissing Value Percentages for {os.path.basename(output_filename)}:")
    print(json.dumps(missing_percentages, indent=4))

    return missing_percentages
def main():
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Process GEP sensor data.')
    parser.add_argument('--year', type=str, choices=['2023', '2024', 'both'], default='both',
                        help='Year to process (2023, 2024, or both)')
    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Extract compressed data
    extract_zip()

    # Define file mappings for 2023
    file_groups_2023 = {
        "Zone 1": {
            "Jan 23_Jun 23": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 1 - Humidity - Jan 23_Jun 23"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 1 - Temperature - Jan 23_Jun 23"),
            },
            "Jun 23_Dec 23": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 1 - Humidity - Jun 23_Dec 23"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 1 - Temperature - Jun 23_Dec 23"),
            }
        },
        "Zone 2": {
            "Jan 23_Jun 23": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 2 - Humidity - Jan 23_Jun 23"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 2 - Temperature - Jan 23_Jun 23"),
            },
            "Jun 23_Dec 23": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 2 - Humidity - Jun 23_Dec 23"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 2 - Temperature - Jun 23_Dec 23"),
            }
        }
    }

    # Define file mappings for 2024
    file_groups_2024 = {
        "Zone 1": {
            "Jan 24_Jun 24": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 1 - Humidity - Jan 24_Jun 24"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 1 - Temperature - Jan 24_Jun 24"),
            }
        },
        "Zone 2": {
            "Jan 24_Jun 24": {
                "humidity": os.path.join(RAW_DATA_DIR, "Zone 2 - Humidity - Jan 24_Jun 24"),
                "temperature": os.path.join(RAW_DATA_DIR, "Zone 2 - Temperature - Jan 24_Jun 24"),
            }
        }
    }

    # Process 2023 data if requested
    if args.year in ['2023', 'both']:
        print("\n" + "=" * 50)
        print("PROCESSING 2023 DATA")
        print("=" * 50)

        # Load outside data file path for 2023
        outside_data_file_2023 = os.path.join(RAW_DATA_DIR, "Mo-BeG_2023_15min.xlsx")

        # Load and process 2023 zone data
        zone_data_dict_2023 = load_zone_data(file_groups_2023)

        # Before processing zones
        calculate_and_save_missing_percentages(
            zone_data_dict_2023['Zone 1'],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone1_before_sync_2023.json")
        )
        calculate_and_save_missing_percentages(
            zone_data_dict_2023['Zone 2'],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone2_before_sync_2023.json")
        )

        # Process 2023 zone data
        process_zone_data(zone_data_dict_2023, outside_data_file_2023,"_23")


    # Process 2024 data if requested
    if args.year in ['2024', 'both']:
        print("\n" + "=" * 50)
        print("PROCESSING 2024 DATA")
        print("=" * 50)

        # Load outside data file path for 2024
        outside_data_file_2024 = os.path.join(RAW_DATA_DIR, "Mo-BeG_2024_Jan-Jun_1min.xlsx")

        # Load and process 2024 zone data
        zone_data_dict_2024 = load_zone_data(file_groups_2024)

        # Before processing zones
        calculate_and_save_missing_percentages(
            zone_data_dict_2024['Zone 1'],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone1_before_sync_2024.json")
        )
        calculate_and_save_missing_percentages(
            zone_data_dict_2024['Zone 2'],
            os.path.join(PROCESSED_DATA_DIR, "missing_percentages_zone2_before_sync_2024.json")
        )

        # Process 2024 zone data
        process_zone_data(zone_data_dict_2024, outside_data_file_2024, "_24")


    print("\n" + "=" * 50)
    print("DATA PROCESSING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()