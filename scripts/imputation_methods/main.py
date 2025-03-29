import argparse
import json
import os
from pathlib import Path
import logging
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from data.loader import DataLoader
from imputation.statistical import StatisticalImputer
from imputation.interpolation import InterpolationImputer
from imputation.ml_methods import MLImputer
from imputation.custom import CustomImputer
from evaluation.metrics import evaluate_imputation
from config import Config
from scripts.GEP_Json_Data_To_CSV import calculate_and_save_missing_percentages
from visualization.plotting import ImputationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Time series data imputation')
    parser.add_argument('--data_path', type=str, default= '../../data/processed/imputed_data_Zone_2_23.csv',
                      help='Path to input CSV file')
    parser.add_argument('--logs_path', type=str, default= './logs',
                      help='Path to save execution logs')
    parser.add_argument('--output_path', type=str,
                        help='Path to save imputed data', default= '../../data/processed/custom_imputation/2023/spline/spline_imputed_2023_zone_2.csv')
    parser.add_argument('--methods', nargs='+', default=['custom'],
                        choices=['mean', 'median', 'locf', 'nocb', 'linear',
                                 'spline', 'knn', 'arima', 'custom'],
                        help='Imputation methods to use')
    parser.add_argument('--columns', nargs='+',
                        help='Columns to impute (default: inside_temperature, inside_humidity)')
    return parser.parse_args()


def setup_logging(logs_path: str):
    log_path = Path(logs_path)  / 'imputation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# Directory setup
processed_data_dir = "../../data/processed/"
plot_dir = os.path.join(processed_data_dir, "quality_plots_after_imputation")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(os.path.join(plot_dir, "detailed_gaps"), exist_ok=True)

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
    print("Columns in aligned_data:", aligned_data.columns)

    # Create year-month column for grouping
    aligned_data['year_month'] = aligned_data['time'].dt.to_period('M')

    print("Missing values in imputed_data based on patterns:",aligned_data[['temperature', 'humidity']].isna().sum())  # Count NaN values
    print("-1 values in imputed_data based on patterns:",(aligned_data[['temperature', 'humidity']] == -1).sum())  # Count -1 values

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
    print("Columns in gap_df:", gap_df.columns)

    # Plot gaps for each zone and month
    if not gap_df.empty:
        for (zone, month), group in gap_df.groupby(['zone', 'year_month']):
            if group.empty:
                # Create an empty figure with message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'No gaps detected for {zone} - {month.strftime("%B %Y")}',
                    fontsize=14, ha='center', va='center', color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)

                # Save the empty plot
                filename = f"no_gaps_{zone.replace(' ', '_')}_{month.strftime('%Y-%m')}_{year_suffix}.png"
                plt.savefig(os.path.join(plot_dir, "detailed_gaps_custom_imputation", filename), dpi=300, bbox_inches='tight')
                plt.close()
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
            plt.savefig(os.path.join(plot_dir, "detailed_gaps_custom_imputation", filename), dpi=300, bbox_inches='tight')
            plt.close()

    return gap_df



def run_imputation(data: pd.DataFrame, method: str, columns: List[str],
                   config: Config) -> pd.DataFrame:
    if method in ['mean', 'median', 'locf', 'nocb']:
        imputer = StatisticalImputer()
        return imputer.impute(data, columns, method)
    elif method in ['linear', 'spline']:
        imputer = InterpolationImputer()
        return imputer.impute(data, columns, method)
    elif method in ['knn', 'arima']:
        imputer = MLImputer(n_neighbors=config.knn_neighbors,
                            arima_order=config.arima_order)
        return imputer.impute(data, columns, method)
    elif method == 'custom':
        imputer = CustomImputer(seasonal_period=config.seasonal_period)
        return imputer.impute(data, columns)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    args = parse_args()
    setup_logging(args.logs_path)
    logger = logging.getLogger(__name__)

    try:
        config = Config()
        if args.columns:
            config.columns_with_nan = args.columns

        loader = DataLoader()
        data = loader.load_data(args.data_path)
        processed_data = loader.preprocess_data(data, config.columns_with_nan)

        ImputationVisualizer.plot_missing_patterns(
            data=processed_data,
            columns=['temperature', 'humidity'],
            output_dir='./visualization/pictures'
        )

        results = {}
        evaluations = {}

        # Run selected imputation methods
        for method in args.methods:
            logger.info(f"Running {method} imputation")
            results[method] = run_imputation(processed_data, method,
                                             config.columns_with_nan, config)
            evaluations[method] = evaluate_imputation(
                processed_data, results[method], config.columns_with_nan, method
            )

            # visualize the results
            ImputationVisualizer.plot_distributions(processed_data, results[method], config.columns_with_nan, method)
            # This preserves all columns while only replacing NA values in the specified columns
            for col in config.columns_with_nan:
                if col in results[method].columns:
                    results[method][col] = results[method][col].replace(pd.NA, -1)
            analyze_and_visualize_gaps(results[method], year_suffix="23_zone_2_custom_imputation")
            calculate_and_save_missing_percentages(
                results[method],
                os.path.join(processed_data_dir + "custom_imputation/2023", "missing_percentages_zone1_after_custom_imputation_Zone_2.json")
            )

        output_path = Path(args.output_path)
        # Determine the best method by minimizing the differences in mean, median, and variance
        best_method = min(
            evaluations.items(),
            key=lambda x: sum(
                sum(diff.get('mean_diff', 0) + diff.get('median_diff', 0) + diff.get('variance_diff', 0)
                    for diff in col_eval.values())
                for col_eval in x[1].values()
                if isinstance(col_eval, dict)
            )
        )[0]

        results[best_method].to_csv(output_path, index=False)

        eval_path = config.base_path +  config.evaluation_path + '/evaluation_metrics.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluations, f, indent=4)

        logger.info(f"Best performing method: {best_method}")
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Evaluation metrics saved to {eval_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
