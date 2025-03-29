from typing import List
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from .base import BaseImputer
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import logging

logger = logging.getLogger(__name__)


class CustomImputer(BaseImputer):
    def __init__(self, seasonal_period: int = 1440, max_window: int = 30):
        self.seasonal_period = seasonal_period
        self.max_window = max_window
        self.weather_columns = [
            'GHI_corr_Avg', 'DNI_corr_Avg', 'DHI_corr_Avg', 'Tair_Avg',
            'RH_Avg', 'BP_CS100_Avg', 'WS_Avg', 'WD_Avg', 'WD_Std', 'Rain_mm_Tot'
        ]

    def impute(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Perform custom imputation for each column with final NaN cleanup.

        Args:
            data: Input DataFrame
            columns: Columns to impute

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        imputed = data.copy()

        try:
            for col in columns:
                imputed = self._impute_column(imputed, 'zone', 'time', col)

            logger.info("Completed custom imputation")
            return imputed

        except Exception as e:
            logger.error(f"Error during custom imputation: {e}")
            raise

    def _impute_column(self, data: pd.DataFrame, zone_column: str,
                       time_column: str, value_column: str) -> pd.DataFrame:
        """Impute a single column using the custom method."""
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        data = data.sort_values(by=[zone_column, time_column])

        processed_zones = []

        try:
            for zone, zone_data in data.groupby(zone_column):
                logger.info(f"Processing zone: {zone}, column: {value_column}")
                zone_data = zone_data.copy()
                zone_data['gap'] = zone_data[value_column].isna()
                zone_data['gap_id'] = (zone_data['gap'] != zone_data['gap'].shift()).cumsum()

                total_gaps = zone_data['gap_id'].nunique()
                for gap_id, gap_data in zone_data.groupby('gap_id'):
                    if not gap_data['gap'].iloc[0]:
                        continue

                    gap_mask = (zone_data['gap_id'] == gap_id) & zone_data['gap']
                    start_time = gap_data[time_column].iloc[0]
                    end_time = gap_data[time_column].iloc[-1]
                    gap_duration = (end_time - start_time).total_seconds() / 60

                    logger.info(f"Processing gap {gap_id}/{total_gaps} in {zone}, duration: {gap_duration} minutes")

                    try:
                        imputed_values = self._impute_gap(
                            zone_data, gap_data, gap_mask, gap_duration, time_column, value_column)
                        if imputed_values is not None:
                            zone_data.loc[gap_mask, value_column] = imputed_values
                    except Exception as e:
                        logger.error(f"Error imputing gap {gap_id} in {zone}: {e}")
                        # Fallback to medium gap interpolation
                        zone_data.loc[gap_mask, value_column] = self._interpolate_medium_gap(
                            zone_data, gap_data, time_column, value_column)

                zone_data = zone_data.drop(columns=['gap', 'gap_id'])
                processed_zones.append(zone_data)

            return pd.concat(processed_zones)
        except Exception as e:
            logger.error(f"Error in _impute_column: {e}")
            raise

    def _impute_gap(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                    gap_mask: pd.Series, gap_duration: float,
                    time_column: str, value_column: str) -> pd.Series | None:
        """Impute values for a single gap."""
        if gap_duration <= 15:
            return self._interpolate_short_gap(zone_data, gap_data, time_column, value_column)
        elif 15 < gap_duration <= 2880: # 2 days window for medium gap ( for 2024 data)
            return self._interpolate_medium_gap(zone_data, gap_data, time_column, value_column)
        else:
            return None

    @staticmethod
    def _interpolate_short_gap(zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                               time_column: str, value_column: str) -> pd.Series:
        """Linear interpolation for short gaps."""
        temp_series = None
        try:
            temp_series = pd.Series(zone_data[value_column].values,
                                    index=zone_data[time_column])
            filled_values = temp_series.interpolate(method='linear', limit=15)  # Limit interpolation
            return filled_values[gap_data[time_column]].values
        except Exception as e:
            logger.error(f"Error in short gap interpolation: {e}")
            # Return forward fill as absolute fallback
            return temp_series.ffill()[gap_data[time_column]].values

    @staticmethod
    def _interpolate_medium_gap(zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                                time_column: str, value_column: str) -> pd.Series:
        """Time-based spline interpolation for medium gaps."""
        temp_series = None
        try:
            temp_series = pd.Series(zone_data[value_column].values,
                                    index=zone_data[time_column])
            filled_values = temp_series.interpolate(method='spline', order=3, limit=2880)
            return filled_values[gap_data[time_column]].values
        except Exception as e:
            logger.error(f"Error in medium gap spline interpolation: {e}")
            # Return forward fill as absolute fallback
            return temp_series.ffill()[gap_data[time_column]].values

    def _impute_long_gap_enhanced(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                                  time_column: str, value_column: str) -> pd.Series:
        """
        Enhanced long gap imputation using pattern matching based on environmental data,
        time of day, and preserving trends.

        Args:
            zone_data: Complete zone dataset
            gap_data: Data for the specific gap
            time_column: Name of the time column
            value_column: Name of the value column to impute

        Returns:
            Imputed values for the gap
        """
        gap_start = gap_data[time_column].iloc[0]
        gap_end = gap_data[time_column].iloc[-1]
        gap_duration = (gap_end - gap_start).total_seconds() / (24 * 60 * 60)  # Convert to days

        logger.info(f"Imputing long gap of {gap_duration} days from {gap_start} to {gap_end}")

        # Preprocess environmental data - fill NaN values
        zone_data_processed = self._preprocess_environmental_data(zone_data.copy())

        # Get data before and after the gap for trend analysis
        before_gap = zone_data_processed[(zone_data_processed[time_column] < gap_start) &
                                         ~zone_data_processed[value_column].isna()].tail(1440)  # Last day before gap
        after_gap = zone_data_processed[(zone_data_processed[time_column] > gap_end) &
                                        ~zone_data_processed[value_column].isna()].head(1440)  # First day after gap

        # Find similar patterns based on environmental data and hour of day
        return self._find_similar_patterns(zone_data_processed, gap_data, before_gap, after_gap, time_column,
                                           value_column)

    def _preprocess_environmental_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess environmental data by filling NaN values using linear interpolation.

        Args:
            data: DataFrame containing environmental data columns

        Returns:
            DataFrame with NaN values filled
        """
        # Identify environmental columns in the data
        env_columns = [col for col in self.weather_columns if col in data.columns]

        if not env_columns:
            logger.warning("No environmental data columns found for preprocessing")
            return data

        # Make a copy to avoid modifying the original
        processed_data = data.copy()

        # Process each zone separately to maintain zone boundaries
        for zone, zone_data in processed_data.groupby('zone'):
            zone_mask = (processed_data['zone'] == zone)

            # Sort by time for proper interpolation
            zone_sorted = zone_data.sort_values(by='time')

            # Process each environmental column
            for col in env_columns:
                # Check if column has NaN values
                if zone_sorted[col].isna().any():
                    # Apply linear interpolation first
                    filled_values = zone_sorted[col].interpolate(method='linear', limit_direction='both')

                    # If any NaN values remain (e.g., at boundaries), use nearest valid value
                    if filled_values.isna().any():
                        filled_values = filled_values.fillna(method='ffill').fillna(method='bfill')

                        # If still NaNs (happens if entire column is NaN), use column mean or 0
                        if filled_values.isna().any():
                            column_mean = filled_values.mean()
                            if np.isnan(column_mean):
                                logger.warning(f"Column {col} for zone {zone} is all NaN, filling with 0")
                                filled_values = filled_values.fillna(0)
                            else:
                                filled_values = filled_values.fillna(column_mean)

                    # Update the processed data
                    processed_data.loc[zone_mask, col] = filled_values.values

        logger.info(f"Preprocessed environmental data, filled NaN values in {len(env_columns)} columns")
        return processed_data

    def _find_similar_patterns(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                               before_gap: pd.DataFrame, after_gap: pd.DataFrame,
                               time_column: str, value_column: str) -> pd.Series:
        """
        Find similar patterns based on environmental variables and hour of day.
        Then adjust to match the trend between before and after gap data.
        """
        gap_start = gap_data[time_column].iloc[0]
        gap_end = gap_data[time_column].iloc[-1]
        gap_length = len(gap_data)

        # Determine search window - use more adaptive approach
        search_start = min(gap_start - pd.Timedelta(days=60), zone_data[time_column].min())
        search_end = max(gap_end + pd.Timedelta(days=60), zone_data[time_column].max())

        # Filter for non-gap data within search window
        search_data = zone_data[
            (zone_data[time_column] >= search_start) &
            (zone_data[time_column] <= search_end) &
            ~zone_data[value_column].isna()
            ]

        # If not enough data found, extend search to all available data
        if len(search_data) < gap_length * 2:
            search_start = zone_data[time_column].min()
            search_end = zone_data[time_column].max()
            search_data = zone_data[
                ((zone_data[time_column] < gap_start) | (zone_data[time_column] > gap_end)) &
                ~zone_data[value_column].isna()
                ]

        # Extract environmental features for the gap
        env_columns = [col for col in self.weather_columns if col in zone_data.columns]

        # If we don't have environmental data available, use hour of day and month
        if not env_columns:
            logger.warning("No environmental data available, using only time-based matching")

        # For each timestamp in the gap, find similar timestamps in the search data
        imputed_values = []

        # Group gap data by day and hour to build patterns
        gap_data = gap_data.copy()
        gap_data['hour'] = gap_data[time_column].dt.hour
        gap_data['day'] = gap_data[time_column].dt.day_of_year
        gap_data['month'] = gap_data[time_column].dt.month
        gap_data['dayofweek'] = gap_data[time_column].dt.dayofweek

        # Similarly prepare search data
        search_data = search_data.copy()
        search_data['hour'] = search_data[time_column].dt.hour
        search_data['day'] = search_data[time_column].dt.day_of_year
        search_data['month'] = search_data[time_column].dt.month
        search_data['dayofweek'] = search_data[time_column].dt.dayofweek

        # Process each day in the gap
        for day, day_gap_data in gap_data.groupby(gap_data[time_column].dt.date):
            day_imputed = []

            # Process each hour in the day
            for hour, hour_gap_data in day_gap_data.groupby('hour'):
                # Find similar hours in the search data
                # Find similar hours in the search data
                similar_hours = search_data[
                    (search_data['hour'] == hour) &
                    (search_data['month'] == hour_gap_data['month'].iloc[0])
                    ]

                if len(similar_hours) < 10:  # Need more data points for robust pattern
                    # Try adjacent months
                    month = hour_gap_data['month'].iloc[0]
                    adjacent_months = [max(1, month - 1), min(12, month + 1)]
                    similar_hours = search_data[
                        (search_data['hour'] == hour) &
                        (search_data['month'].isin([month] + adjacent_months))
                        ]

                    if len(similar_hours) < 10:
                        # If still not enough, then just match hour
                        similar_hours = search_data[search_data['hour'] == hour]

                # Calculate similarity scores based on environmental variables
                similarity_scores = []

                for _, similar_hour_data in similar_hours.groupby(similar_hours[time_column].dt.date):
                    # Skip if not enough data points
                    if len(similar_hour_data) < len(hour_gap_data) / 2:
                        continue

                    # Calculate similarity score based on environmental variables
                    score = 0
                    count = 0

                    for env_col in env_columns:
                        # Get values from gap data surroundings (before and after gap)
                        env_before = before_gap[env_col].mean() if not before_gap.empty else None
                        env_after = after_gap[env_col].mean() if not after_gap.empty else None

                        # Get values from candidate pattern
                        pattern_env = similar_hour_data[env_col].mean()

                        # Calculate score if we have valid data
                        if env_before is not None and not np.isnan(env_before):
                            score += abs(pattern_env - env_before)
                            count += 1
                        if env_after is not None and not np.isnan(env_after):
                            score += abs(pattern_env - env_after)
                            count += 1

                    # Also consider time proximity as a factor
                    time_diff = abs((similar_hour_data[time_column].iloc[0] - hour_gap_data[time_column].iloc[0]).days)
                    time_score = min(1.0, time_diff / 365)  # Normalize by year

                    # Consider day of week similarity (0 if same day, higher if different)
                    dow_diff = abs(similar_hour_data['dayofweek'].iloc[0] - hour_gap_data['dayofweek'].iloc[0])
                    dow_score = min(1.0, dow_diff / 7)  # Normalize by week

                    # Calculate final score (lower is better)
                    # Give more weight to time-based factors if environmental data is sparse
                    if count > 0:
                        final_score = (score / count) + time_score + dow_score
                    else:
                        final_score = time_score + dow_score * 2  # Time factors matter more when no env data

                    similarity_scores.append({
                        'data': similar_hour_data,
                        'score': final_score
                    })

                # Sort by score (lower is better)
                similarity_scores.sort(key=lambda x: x['score'])

                # Take top 3 similar patterns and calculate weighted average
                # If fewer than 3 available, use what we have
                top_patterns = similarity_scores[:3] if len(similarity_scores) >= 3 else similarity_scores

                if not top_patterns:
                    # This should be unlikely now given our enhanced matching above,
                    # but as a safeguard, add another fallback
                    seasonal_data = zone_data[
                        (zone_data[time_column].dt.hour == hour) &
                        ~zone_data[value_column].isna()
                        ]

                    if len(seasonal_data) > 0:
                        # Use hour-based data
                        hour_values = [seasonal_data[value_column].mean()] * len(hour_gap_data)
                        day_imputed.extend(hour_values)
                    else:
                        # Last resort: medium gap interpolation
                        fallback_values = self._interpolate_medium_gap(
                            zone_data, hour_gap_data, time_column, value_column
                        )
                        day_imputed.extend(fallback_values)
                    continue

                # Calculate weights based on inverse of score
                # Add a small constant to avoid division by zero
                total_weight = sum(1 / (p['score'] + 0.1) for p in top_patterns)

                # For each minute in the gap hour, calculate weighted average
                for _, gap_minute in hour_gap_data.iterrows():
                    gap_minute_time = gap_minute[time_column]
                    gap_minute_of_hour = gap_minute_time.minute

                    weighted_sum = 0
                    total_minute_weight = 0

                    for pattern in top_patterns:
                        pattern_data = pattern['data']
                        pattern_weight = 1 / (pattern['score'] + 0.1) / total_weight

                        # Find matching minute in pattern
                        pattern_minute = pattern_data[pattern_data[time_column].dt.minute == gap_minute_of_hour]

                        if not pattern_minute.empty and not np.isnan(pattern_minute[value_column].iloc[0]):
                            weighted_sum += pattern_minute[value_column].iloc[0] * pattern_weight
                            total_minute_weight += pattern_weight

                    if total_minute_weight > 0:
                        # We found matching minutes in patterns
                        day_imputed.append(weighted_sum / total_minute_weight)
                    else:
                        # Fallback for this specific minute
                        closest_values = []
                        for pattern in top_patterns:
                            pattern_data = pattern['data']
                            if not pattern_data.empty:
                                closest_minute = pattern_data.iloc[
                                    (pattern_data[time_column].dt.minute - gap_minute_of_hour).abs().argsort()[:1]
                                ]
                                if not closest_minute.empty:
                                    closest_values.append(closest_minute[value_column].iloc[0])

                        if closest_values:
                            day_imputed.append(sum(closest_values) / len(closest_values))
                        else:
                            # Ultimate fallback - use nearby values
                            last_value = before_gap[value_column].iloc[-1] if not before_gap.empty else None
                            next_value = after_gap[value_column].iloc[0] if not after_gap.empty else None

                            if last_value is not None and next_value is not None:
                                # Linear interpolation between last value before gap and first value after gap
                                gap_progress = (gap_minute_time - gap_start).total_seconds() / (
                                        gap_end - gap_start).total_seconds()
                                day_imputed.append(last_value + gap_progress * (next_value - last_value))
                            elif last_value is not None:
                                day_imputed.append(last_value)
                            elif next_value is not None:
                                day_imputed.append(next_value)
                            else:
                                # Should rarely happen with our enhanced approach
                                # Use daily average by hour as last resort
                                hour_data = zone_data[zone_data[time_column].dt.hour == hour][value_column]
                                if not hour_data.empty:
                                    day_imputed.append(hour_data.mean())
                                else:
                                    day_imputed.append(zone_data[value_column].mean())

                # Add day's imputed values to overall result
                imputed_values.extend(day_imputed)

        # Ensure we have the correct number of values
        if len(imputed_values) != len(gap_data):
            logger.warning(f"Generated {len(imputed_values)} values but gap has {len(gap_data)} points; adjusting...")

            # Create correctly sized series with appropriate time index
            if len(imputed_values) > len(gap_data):
                # If we generated too many values, downsample to match the gap
                # This is better than simply truncating
                temp_index = pd.date_range(start=gap_data[time_column].iloc[0],
                                           end=gap_data[time_column].iloc[-1],
                                           periods=len(imputed_values))
                temp_series = pd.Series(imputed_values, index=temp_index)
                # Resample to match the actual gap points
                imputed_series = temp_series.reindex(gap_data[time_column], method='nearest')
            else:
                # If we generated too few values, create series and interpolate missing points
                imputed_series = pd.Series(imputed_values,
                                           index=gap_data[time_column][:len(imputed_values)])
                imputed_series = imputed_series.reindex(gap_data[time_column])
                imputed_series = imputed_series.interpolate(method='time')

            imputed_values = imputed_series.values

        # Ensure the imputed values follow the trend between before and after gap
        imputed_values = self._adjust_for_trend(imputed_values, before_gap, after_gap, value_column, gap_length)

        # Verify imputation - make sure values aren't constant
        if not self._verify_imputation(imputed_values, zone_data, value_column):
            logger.warning("Imputation verification failed, trying seasonal approach")

            # Instead of falling back to medium gap interpolation, try a seasonal approach
            seasonal_values = self._seasonal_fallback_imputation(zone_data, gap_data, time_column, value_column)
            if seasonal_values is not None:
                return seasonal_values

            logger.warning("Seasonal approach failed too, falling back to medium gap interpolation")
            return self._interpolate_medium_gap(zone_data, gap_data, time_column, value_column)

        return pd.Series(imputed_values, index=gap_data[time_column])

    def _seasonal_fallback_imputation(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                                      time_column: str, value_column: str) -> pd.Series:
        """
        Impute values based on seasonal patterns when other methods fail.
        This uses hour of day, day of week patterns to impute.
        """
        try:
            # Create a copy of gap data with additional time features
            gap_data = gap_data.copy()
            gap_data['hour'] = gap_data[time_column].dt.hour
            gap_data['dayofweek'] = gap_data[time_column].dt.dayofweek

            # Prepare full dataset with time features
            full_data = zone_data.copy()
            full_data['hour'] = full_data[time_column].dt.hour
            full_data['dayofweek'] = full_data[time_column].dt.dayofweek

            # Get valid data only
            valid_data = full_data[~full_data[value_column].isna()]

            # Initialize result series
            result = pd.Series(index=gap_data[time_column])

            # Try hour-dow means first (most specific)
            hour_dow_means = valid_data.groupby(['hour', 'dayofweek'])[value_column].mean()

            # If hour-dow combinations are too sparse, use just hour means
            hour_means = valid_data.groupby(['hour'])[value_column].mean()

            # Fill in values
            for idx, row in gap_data.iterrows():
                hour = row['hour']
                dow = row['dayofweek']

                # Try hour-dow mean first
                if (hour, dow) in hour_dow_means.index and not np.isnan(hour_dow_means[(hour, dow)]):
                    result[row[time_column]] = hour_dow_means[(hour, dow)]
                # Fall back to hour mean
                elif hour in hour_means.index and not np.isnan(hour_means[hour]):
                    result[row[time_column]] = hour_means[hour]
                # If still no value, use overall mean
                else:
                    result[row[time_column]] = valid_data[value_column].mean()

            # Apply a smoothing filter to avoid abrupt changes
            smoothed_values = result.rolling(window=5, center=True, min_periods=1).mean().values

            return pd.Series(smoothed_values, index=gap_data[time_column])

        except Exception as e:
            logger.error(f"Error in seasonal fallback imputation: {e}")
            return None

    def _adjust_for_trend(self, imputed_values, before_gap, after_gap, value_column, gap_length):
        """Adjust imputed values to follow the trend between before and after gap."""
        # If we have data both before and after the gap
        if not before_gap.empty and not after_gap.empty:
            before_value = before_gap[value_column].iloc[-1]
            after_value = after_gap[value_column].iloc[0]

            # Calculate the trend (slope)
            trend_slope = (after_value - before_value) / gap_length

            # If there's a significant trend, adjust values
            if abs(trend_slope) > 0.01:  # Threshold for considering a trend significant
                # Calculate baseline (average of imputed values)
                baseline = np.mean(imputed_values)

                # Adjust each value to follow the trend while preserving pattern
                for i in range(len(imputed_values)):
                    # Calculate trend component
                    trend_component = before_value + trend_slope * i

                    # Calculate pattern component (deviation from baseline)
                    pattern_component = imputed_values[i] - baseline

                    # Combine trend and pattern
                    imputed_values[i] = trend_component + pattern_component

        # Apply a smoothing filter to avoid abrupt changes
        if len(imputed_values) > 5:
            # Use pandas rolling window to smooth the values
            smoothed_values = pd.Series(imputed_values).rolling(window=5, center=True, min_periods=1).mean().values
            imputed_values = smoothed_values

        return imputed_values

    def _verify_imputation(self, imputed_values, zone_data, value_column):
        # Check for long stretches of constant values instead of all values being constant
        max_constant_stretch = 0
        current_stretch = 1
        for i in range(1, len(imputed_values)):
            if round(imputed_values[i], 3) == round(imputed_values[i - 1], 3):
                current_stretch += 1
            else:
                max_constant_stretch = max(max_constant_stretch, current_stretch)
                current_stretch = 1

        max_constant_stretch = max(max_constant_stretch, current_stretch)

        # Fail if more than half the values are in a constant stretch
        if max_constant_stretch > 0.5 * len(imputed_values):
            logger.warning(f"Imputed values have a constant stretch of {max_constant_stretch}/{len(imputed_values)}")
            return False