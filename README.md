# THERMODSET: Anomaly-Free and Open-Source Dataset for Thermal Modeling

This project handles analysis, imputation, and anomaly detection for temperature and humidity sensor data. It provides tools to preprocess data, detect gaps and anomalies, and impute missing values using various methods including custom pattern-matching algorithms and ML-based approaches.

## Requirements

### Python Dependencies
```
pandas>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
xgboost>=1.5.0
fastdtw>=0.3.0
statsmodels>=0.13.0
```

### System Requirements
- Python 3.10+
- Sufficient storage for data files (at least 5GB recommended)
- At least 8GB RAM for processing larger datasets

## Project Structure

```
├── data/
│   ├── raw/           # Original sensor data
│   ├── processed/     # Cleaned and processed data
│   └── output/        # Final results and visualizations
├── scripts/
│   ├── imputation_methods/    # Different imputation techniques
│   │   ├── ml_based_imputation/    # Machine learning models 
│   │   ├── custom.py           # Custom pattern-based imputation
│   │   └── interpolation.py    # Basic interpolation methods
│   ├── GEP_Json_Data_To_CSV.py    # Initial data loading utilities
│   └── comparison_plots_generator.py   # Create comparative visualizations
└── visualization/    # Tools for creating plots and analysis visuals
```

## Step 1: Data Loading and Preparation

1. Clone the repository and navigate to the project directory

2. Process raw data from JSON to CSV format:
```bash
python scripts/GEP_Json_Data_To_CSV.py --input_dir data/raw --output_dir data/processed
```

3. Perform initial data analysis to identify data quality issues:
```bash
python scripts/imputation_methods/main.py --data_path data/processed/final_merged_data_Zone_1_23.csv --logs_path ./logs
```

This will generate initial visualizations showing gaps in the data and save them to the `data/processed/quality_plots` directory.

## Step 2: Anomaly Detection

Detect anomalies in the temperature and humidity data:

```bash
python scripts/anomaly_detection.py --input data/processed/imputed_data_Zone_1_23.csv --output data/processed/anomalies_Zone_1_23.json
```

This will:
- Identify statistical outliers
- Detect physically impossible values
- Find sensor drift patterns
- Create visualizations of detected anomalies

## Step 3: Data Imputation

### Custom Imputation Method

Apply the pattern-based custom imputation method:

```bash
python scripts/imputation_methods/main.py --data_path data/processed/final_merged_data_Zone_1_23.csv --methods custom --output_path data/processed/custom_imputation/custom_imputed_Zone_1_23.csv
```

This method:
- Detects gap types (short, medium, long)
- Uses pattern matching for long gaps
- Applies trend-aware interpolation
- Preserves daily and seasonal patterns

### ML-Based Imputation

Train ML models and apply them for imputation:

```bash
# First train the models
python scripts/imputation_methods/ml_based_imputation/train_models.py --data data/processed/final_merged_data_Zone_1_23.csv --model_dir models

# Then apply imputation
python scripts/imputation_methods/ml_based_imputation/impute_missing.py --data data/processed/final_merged_data_Zone_1_23.csv --model_dir models --output data/processed/ml_imputed_Zone_1_23.csv
```

This approach:
- Uses XGBoost regression models for temperature and humidity
- Leverages environmental variables as predictors
- Handles different types of gaps effectively

### Generate Comparison Plots

Compare the imputation methods:

```bash
python scripts/comparison_plots_generator.py
```

This generates visual comparisons between the results of imputation methods, showing the gaps distribution before and after such imputation method.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
