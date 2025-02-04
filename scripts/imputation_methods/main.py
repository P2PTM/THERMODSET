import argparse
import json
from pathlib import Path
import logging
from typing import List, Optional

import pandas as pd

from data.loader import DataLoader
from imputation.statistical import StatisticalImputer
from imputation.interpolation import InterpolationImputer
from imputation.ml_methods import MLImputer
from imputation.custom import CustomImputer
from evaluation.metrics import evaluate_imputation
from config import Config
from visualization.plotting import ImputationVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Time series data imputation')
    parser.add_argument('--data_path', type=str, default= '../../data/processed/final_merged_data.csv',
                      help='Path to input CSV file')
    parser.add_argument('--logs_path', type=str, default= './logs',
                      help='Path to save execution logs')
    parser.add_argument('--output_path', type=str,
                        help='Path to save imputed data', default= '../../data/processed/imputedData.csv')
    parser.add_argument('--methods', nargs='+', default=['custom','linear','nocb','median'],
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
            columns=['inside_temperature', 'inside_humidity'],
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
