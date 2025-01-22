from typing import List
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from .base import BaseImputer
import logging

logger = logging.getLogger(__name__)


class CustomImputer(BaseImputer):

    def __init__(self, seasonal_period: int = 1440):
        self.seasonal_period = seasonal_period

    def impute(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Perform custom imputation for each column.

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

        for zone, zone_data in data.groupby(zone_column):
            zone_data = zone_data.copy()
            zone_data['gap'] = zone_data[value_column].isna()
            zone_data['gap_id'] = (zone_data['gap'] != zone_data['gap'].shift()).cumsum()

            for gap_id, gap_data in zone_data.groupby('gap_id'):
                if not gap_data['gap'].iloc[0]:
                    continue

                gap_mask = (zone_data['gap_id'] == gap_id) & zone_data['gap']
                start_time = gap_data[time_column].iloc[0]
                end_time = gap_data[time_column].iloc[-1]
                gap_duration = (end_time - start_time).total_seconds() / 60

                zone_data.loc[gap_mask, value_column] = self._impute_gap(
                    zone_data, gap_data, gap_mask, gap_duration, time_column, value_column)

            zone_data = zone_data.drop(columns=['gap', 'gap_id'])
            processed_zones.append(zone_data)

        return pd.concat(processed_zones)

    def _impute_gap(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                    gap_mask: pd.Series, gap_duration: float,
                    time_column: str, value_column: str) -> pd.Series:
        """Impute values for a single gap."""
        if gap_duration <= 10:
            return self._interpolate_short_gap(zone_data, gap_data, time_column, value_column)
        elif 10 < gap_duration <= 60:
            return self._interpolate_medium_gap(zone_data, gap_data, time_column, value_column)
        else:
            return self._impute_long_gap(zone_data, gap_data, time_column, value_column)

    @staticmethod
    def _interpolate_short_gap(zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                               time_column: str, value_column: str) -> pd.Series:
        """Linear interpolation for short gaps."""
        temp_series = pd.Series(zone_data[value_column].values,
                                index=zone_data[time_column])
        filled_values = temp_series.interpolate(method='linear')
        return filled_values[gap_data[time_column]].values

    @staticmethod
    def _interpolate_medium_gap(zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                                time_column: str, value_column: str) -> pd.Series:
        """Time-based interpolation for medium gaps."""
        temp_series = pd.Series(zone_data[value_column].values,
                                index=zone_data[time_column])
        filled_values = temp_series.interpolate(method='time')
        return filled_values[gap_data[time_column]].values

    def _impute_long_gap(self, zone_data: pd.DataFrame, gap_data: pd.DataFrame,
                         time_column: str, value_column: str) -> pd.Series:
        """Seasonal decomposition for long gaps."""
        valid_data = zone_data[~zone_data[value_column].isna()]

        if len(valid_data) < self.seasonal_period:
            return self._interpolate_medium_gap(zone_data, gap_data, time_column, value_column)

        try:
            temp_series = pd.Series(valid_data[value_column].values,
                                    index=valid_data[time_column])
            decomposed = seasonal_decompose(temp_series, period=self.seasonal_period)
            trend = pd.Series(decomposed.trend, index=temp_series.index)
            trend = trend.ffill().bfill()
            return trend.reindex(gap_data[time_column]).values
        except:
            return self._interpolate_medium_gap(zone_data, gap_data, time_column, value_column)
