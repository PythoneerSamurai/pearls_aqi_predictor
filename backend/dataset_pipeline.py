from datetime import datetime, timedelta
from os import getenv

from dotenv import load_dotenv
from hopsworks import login
from numpy import cos, sin, radians
from openmeteo_requests import Client
from pandas import DataFrame, date_range, to_datetime, merge, Timedelta
from requests_cache import CachedSession
from retry_requests import retry


class DatasetPipeline:
    def __init__(self):
        load_dotenv()

        self._project = login(
            host='eu-west.cloud.hopsworks.ai',
            project='haroons_aqi_predictor',
            api_key_value=getenv("API_KEY"),
        )
        self._fs = self._project.get_feature_store()

        self._features_url = "https://api.open-meteo.com/v1/forecast"
        self._targets_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

        self._latitude = 33.5973
        self._longitude = 73.0479

    def _fetch_data(self, start_date: str, end_date: str) -> tuple[DataFrame, DataFrame]:
        feature_params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "surface_pressure", "precipitation", "rain", "cloud_cover",
                "shortwave_radiation"
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        }

        target_params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": [
                "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                "sulphur_dioxide", "ozone", "us_aqi",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        }

        cache_session = CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = Client(session=retry_session)

        raw_features = openmeteo.weather_api(url=self._features_url, params=feature_params)[0]
        raw_targets = openmeteo.weather_api(url=self._targets_url, params=target_params)[0]

        hourly_raw_features = raw_features.Hourly()
        hourly_raw_targets = raw_targets.Hourly()

        features_data = {
            "datetime": date_range(
                start=to_datetime(hourly_raw_features.Time(), unit="s", utc=True),
                end=to_datetime(hourly_raw_features.TimeEnd(), unit="s", utc=True),
                freq=Timedelta(seconds=hourly_raw_features.Interval()),
                inclusive="left"
            )
        }
        for index, feature in enumerate(feature_params["hourly"]):
            features_data[feature] = hourly_raw_features.Variables(index).ValuesAsNumpy()

        targets_data = {
            "datetime": date_range(
                start=to_datetime(hourly_raw_targets.Time(), unit="s", utc=True),
                end=to_datetime(hourly_raw_targets.TimeEnd(), unit="s", utc=True),
                freq=Timedelta(seconds=hourly_raw_targets.Interval()),
                inclusive="left"
            )
        }
        for index, feature in enumerate(target_params["hourly"]):
            if index == len(target_params["hourly"]) - 1:
                break
            features_data[feature] = hourly_raw_targets.Variables(index).ValuesAsNumpy()

        targets_data["us_aqi"] = hourly_raw_targets.Variables(6).ValuesAsNumpy()

        return DataFrame(features_data), DataFrame(targets_data)

    def _engineer_features(self, df: DataFrame) -> DataFrame:
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Karachi')
        df = df.sort_values('datetime').reset_index(drop=True)

        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18]).astype(int)

        def get_season(month):
            if month in [12, 1, 2]:
                return 0
            elif month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8]:
                return 2
            else:
                return 3

        df['season'] = df['month'].apply(get_season)

        df['wind_u'] = df['wind_speed_10m'] * cos(radians(df['wind_direction_10m']))
        df['wind_v'] = df['wind_speed_10m'] * sin(radians(df['wind_direction_10m']))
        df['is_stagnant'] = (df['wind_speed_10m'] < 2).astype(int)
        df['temp_humidity_product'] = df['temperature_2m'] * df['relative_humidity_2m']

        return df

    def _store_in_feature_store(self, df: DataFrame) -> None:
        aqi_fg = self._fs.get_or_create_feature_group(
            name="aqi_hourly_features",
            version=1,
            description="Hourly AQI features (weather + pollutants + temporal)",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False,
        )
        aqi_fg.insert(df, write_options={"wait_for_job": False})

    def run_hourly_update(self) -> None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        features_df, targets_df = self._fetch_data(start_date, end_date)

        merged_df = merge(features_df, targets_df, on='datetime', how='inner')
        engineered_df = self._engineer_features(merged_df)

        self._store_in_feature_store(engineered_df)

    def run_historical_backfill(self, days: int = 92) -> None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        features_df, targets_df = self._fetch_data(start_date, end_date)

        merged_df = merge(features_df, targets_df, on='datetime', how='inner')
        engineered_df = self._engineer_features(merged_df)

        self._store_in_feature_store(engineered_df)
