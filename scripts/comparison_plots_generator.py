import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import seaborn as sns

# Directory setup
processed_data_dir = "../data/processed/"
plot_comparison_dir = os.path.join(processed_data_dir,
                                   "custom_imputation/visualization/anomaly_free/plots_trends_comparison")
os.makedirs(plot_comparison_dir, exist_ok=True)

# Load datasets for "before" comparison
df_before_2023 = pd.read_csv(os.path.join(processed_data_dir + "custom_imputation/2023/ml_based", "imputed_ml_based_2023_Zone_1.csv"))
df_before_2024 = pd.read_csv(os.path.join(processed_data_dir, "forecasted_data_Zone_1_24.csv"))

# Load datasets for "after" comparison
df_after_2023 = pd.read_csv(
    os.path.join(processed_data_dir + "custom_imputation/2023/anomalies_free", "anomalies_free_2023_1.csv"))
df_after_2024 = pd.read_csv(
    os.path.join(processed_data_dir + "custom_imputation/2023/anomalies_free", "anomalies_free_2024_1.csv"))

# Convert 'time' column to datetime for all datasets
df_before_2023['time'] = pd.to_datetime(df_before_2023['time'])
df_before_2024['time'] = pd.to_datetime(df_before_2024['time'])
df_after_2023['time'] = pd.to_datetime(df_after_2023['time'])
df_after_2024['time'] = pd.to_datetime(df_after_2024['time'])

# Filter out June 2024 data (keep only Jan-May 2024)
df_before_2024 = df_before_2024[df_before_2024['time'].dt.month < 6]
df_after_2024 = df_after_2024[df_after_2024['time'].dt.month < 6]

# Merge the datasets for "before" and "after"
df_before = pd.concat([df_before_2023, df_before_2024])
df_after = pd.concat([df_after_2023, df_after_2024])

# Fill NA values with -1
df_before = df_before.fillna(-1)
df_after = df_after.fillna(-1)

# Resample to daily data
daily_before = df_before.resample('D', on='time').mean(numeric_only=True).reset_index()
daily_after = df_after.resample('D', on='time').mean(numeric_only=True).reset_index()

# Add a label column to distinguish before and after data
daily_before['label'] = 'Before'
daily_after['label'] = 'After'

# Combine before and after data for boxplots
combined_data = pd.concat([df_before.assign(label='Before'), df_after.assign(label='After')])

# Extract month and year from the 'time' column
combined_data['month'] = combined_data['time'].dt.month_name()
combined_data['year'] = combined_data['time'].dt.year
combined_data['month_year'] = combined_data['time'].dt.strftime('%b %Y')

# Create comparison plots
# Temperature Trend Comparison
plt.figure(figsize=(15, 6))
plt.plot(daily_before['time'], daily_before['temperature'], color='red', label='Before Temperature')
plt.plot(daily_after['time'], daily_after['temperature'], color='orange', label='After Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trend Comparison (Before vs After) - Jan 2023 to May 2024')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(plot_comparison_dir, "temperature_trend_comparison.png"), dpi=300)
plt.close()

# Humidity Trend Comparison
plt.figure(figsize=(15, 6))
plt.plot(daily_before['time'], daily_before['humidity'], color='blue', label='Before Humidity')
plt.plot(daily_after['time'], daily_after['humidity'], color='cyan', label='After Humidity')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trend Comparison (Before vs After) - Jan 2023 to May 2024')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.savefig(os.path.join(plot_comparison_dir, "humidity_trend_comparison.png"), dpi=300)
plt.close()


# Create a function to generate boxplots by month-year
def create_boxplots_by_month_year(data, var_name, color_before, color_after, title_suffix, filename_prefix):
    plt.figure(figsize=(18, 10))
    sns.set(style="whitegrid")

    # Get unique month-year combinations in chronological order
    month_years = sorted(data['month_year'].unique(),
                         key=lambda x: pd.to_datetime(x, format='%b %Y'))

    # Create boxplots for "Before" data
    plt.subplot(2, 1, 1)
    sns.boxplot(x='month_year', y=var_name,
                data=data[data['label'] == 'Before'],
                order=month_years,
                palette=color_before)
    plt.title(f'{title_suffix} Boxplot (Before) - Jan 2023 to May 2024')
    plt.ylabel(f'{title_suffix} ' + ('(°C)' if var_name == 'temperature' else '(%)'))
    plt.xlabel('')
    plt.xticks(rotation=45)

    # Create boxplots for "After" data
    plt.subplot(2, 1, 2)
    sns.boxplot(x='month_year', y=var_name,
                data=data[data['label'] == 'After'],
                order=month_years,
                palette=color_after)
    plt.title(f'{title_suffix} Boxplot (After) - Jan 2023 to May 2024')
    plt.ylabel(f'{title_suffix} ' + ('(°C)' if var_name == 'temperature' else '(%)'))
    plt.xlabel('Month-Year')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_comparison_dir, f"{filename_prefix}_boxplot_comparison.png"), dpi=300)
    plt.close()


# Generate boxplots for temperature and humidity by month-year
create_boxplots_by_month_year(
    combined_data,
    'temperature',
    'Reds',
    'Oranges',
    'Temperature',
    'temperature'
)

create_boxplots_by_month_year(
    combined_data,
    'humidity',
    'Blues',
    'Greens',
    'Humidity',
    'humidity'
)

print("All comparison plots have been generated and saved in the 'plots_trends_comparison' folder!")