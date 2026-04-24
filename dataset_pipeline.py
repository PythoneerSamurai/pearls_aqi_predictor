from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from pandas import DataFrame
from retry_requests import retry


class DatasetPipeline:
    def __init__(self, past_days: int = 92):
        self._features_url = "https://api.open-meteo.com/v1/forecast"
        self._targets_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

        self._feature_params = {
            "latitude": 33.5973,
            "longitude": 73.0479,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
                       "surface_pressure", "precipitation", "cloud_cover"],
            "timezone": "auto",
            "past_days": past_days,
            "forecast_days": 1,
        }
        self._target_params = {
            "latitude": 33.5973,
            "longitude": 73.0479,
            "hourly": ["pm10", "pm2_5", "ozone"],
            "current": "us_aqi",
            "timezone": "auto",
            "past_days": past_days,
            "forecast_days": 1,
        }

    def _feature_extractor(
            self,
            raw_features: dict[str, Any],
            raw_targets: dict[str, Any],
    ) -> tuple(DataFrame, DataFrame):

        hourly_raw_features = raw_features.Hourly()
        hourly_raw_targets = raw_targets.Hourly()

        hourly_features = {"date": pd.date_range(
            start=pd.to_datetime(hourly_raw_features.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly_raw_features.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly_raw_features.Interval()),
            inclusive="left"
        )}
        hourly_targets = {"date": pd.date_range(
            start=pd.to_datetime(hourly_raw_targets.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly_raw_targets.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly_raw_targets.Interval()),
            inclusive="left"
        )}

        for index, feature in enumerate(self._feature_params["hourly"]):
            hourly_features[feature] = hourly_raw_features.Variables(index).ValuesAsNumpy()

        for index, target in enumerate(self._target_params["hourly"]):
            hourly_targets[target] = hourly_raw_targets.Variables(index).ValuesAsNumpy()

        hourly_features = pd.DataFrame(data=hourly_features)
        hourly_targets = pd.DataFrame(data=hourly_targets)

        return (hourly_features, hourly_targets)

    def historical_data_backfill(self) -> None:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        raw_features = openmeteo.weather_api(url=self._features_url, params=self._feature_params)[0]
        raw_targets = openmeteo.weather_api(url=self._targets_url, params=self._target_params)[0]

        features, targets = self._feature_extractor(raw_features, raw_targets)


DatasetPipeline().historical_data_backfill()
