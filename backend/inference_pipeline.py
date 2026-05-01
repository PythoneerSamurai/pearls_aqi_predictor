import os
from datetime import datetime, timedelta
from typing import Any

import hopsworks
import joblib
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from pandas.core.interchange.dataframe_protocol import DataFrame
from retry_requests import retry


class InferencePipeline:
    def __init__(self):
        load_dotenv()

        self._project = hopsworks.login(
            host='eu-west.cloud.hopsworks.ai',
            project='haroons_aqi_predictor',
            api_key_value=os.getenv("API_KEY")
        )
        self._fs = self._project.get_feature_store()
        self._mr = self._project.get_model_registry()

        self._latitude = 33.5973
        self._longitude = 73.0479

        self._weather_features_url = "https://api.open-meteo.com/v1/forecast"
        self._aqi_features_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def _fetch_forecast_data(self, days_ahead: int = 3) -> DataFrame:
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        weather_feature_params = {
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
        aqi_feature_params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "hourly": [
                "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                "sulphur_dioxide", "ozone",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        }

        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        raw_weather_features = openmeteo.weather_api(url=self._weather_features_url, params=weather_feature_params)[0]
        raw_aqi_features = openmeteo.weather_api(url=self._aqi_features_url, params=aqi_feature_params)[0]

        hourly_raw_features = raw_weather_features.Hourly()
        hourly_raw_targets = raw_aqi_features.Hourly()

        features_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly_raw_features.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly_raw_features.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly_raw_features.Interval()),
                inclusive="left"
            )
        }
        for index, feature in enumerate(weather_feature_params["hourly"]):
            features_data[feature] = hourly_raw_features.Variables(index).ValuesAsNumpy()

        for index, feature in enumerate(aqi_feature_params["hourly"]):
            features_data[feature] = hourly_raw_targets.Variables(index).ValuesAsNumpy()

        df = pd.DataFrame(features_data)
        
        return df

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

        df['wind_u'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        df['wind_v'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))
        df['is_stagnant'] = (df['wind_speed_10m'] < 2).astype(int)
        df['temp_humidity_product'] = df['temperature_2m'] * df['relative_humidity_2m']

        return df

    def _load_models(self) -> dict[str, Any]:
        model_names = ["random_forest", "gradient_boosting", "svr", "knn", "xgboost"]
        models = {}

        for model_name in model_names:
            model = self._mr.get_models(model_name, version=int(os.getenv("MODEL_VERSION")))
            models.update({model_name: model})
            model_dir = model.download(local_path="temp")
            loaded_model = joblib.load(f"{model_dir}/{model_name}.pkl")
            models[model_name] = loaded_model
            os.remove(f"{model_dir}/{model_name}.pkl")

        return models

    def predict(self, days_ahead: int = 3) -> DataFrame:
        forecast_df = self._fetch_forecast_data(days_ahead)
        forecast_df = self._engineer_features(forecast_df)

        models = self._load_models()

        feature_cols = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "surface_pressure", "precipitation", "rain", "cloud_cover",
            "shortwave_radiation", "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_rush_hour",
            "season", "wind_u", "wind_v", "is_stagnant", "temp_humidity_product"
        ]

        X = forecast_df[feature_cols].fillna(forecast_df[feature_cols].mean())

        predictions = {
            'datetime': forecast_df['datetime']
        }

        for model_name, model in models.items():
            predictions[f'{model_name}_prediction'] = model.predict(X)

        model_preds = [predictions[f'{name}_prediction'] for name in models.keys()]
        predictions['ensemble_prediction'] = np.mean(model_preds, axis=0)

        predictions_df = pd.DataFrame(predictions)

        return predictions_df

    def get_daily_summary(self, predictions_df) -> DataFrame:
        predictions_df['date'] = predictions_df['datetime'].dt.date

        daily_summary = predictions_df.groupby('date').agg({
            'random_forest_prediction': 'mean',
            'gradient_boosting_prediction': 'mean',
            'svr_prediction': 'mean',
            'knn_prediction': 'mean',
            'xgboost_prediction': 'mean',
            'ensemble_prediction': 'mean'
        }).round(2)

        return daily_summary
