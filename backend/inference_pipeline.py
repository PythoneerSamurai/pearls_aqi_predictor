# Imports.
from datetime import datetime, timedelta
from os import getenv
import logging  # For generating logs. Useful for debugging.

from dotenv import load_dotenv
from hopsworks import login
from numpy import array, mean, cos, sin, radians  # For feature engineering and handling missing values.
from openmeteo_requests import Client
from pandas import DataFrame, date_range, to_datetime, Timedelta
from requests_cache import CachedSession
from retry_requests import retry

# Setting up logger to save logs to stdout.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Main pipeline class.
class InferencePipeline:
    def __init__(self):
        logger.info("Initializing InferencePipeline")
        load_dotenv()

        try:
            # Logging into the project.
            self._project = login(
                host="eu-west.cloud.hopsworks.ai",
                project="haroons_aqi_predictor",
                api_key_value=getenv("API_KEY")
            )
            logger.info("Successfully connected to Hopsworks project")

            self._ms = self._project.get_model_serving()  # For getting predictions from models.
            logger.debug("Retrieved feature store and model registry")

            # Rawalpindi coordinates.
            self._latitude = 33.5973
            self._longitude = 73.0479
            logger.info(f"Location set to: ({self._latitude}, {self._longitude})")

            # Same thing as the dataset pipeline. Two APIs, one for weather data, the other for pollutant data.
            self._weather_features_url = "https://api.open-meteo.com/v1/forecast"
            self._aqi_features_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

            # Names of the trained and deployed models.
            self._model_names = ["random_forest", "gradient_boosting", "svr", "knn", "xgboost"]

        except Exception as e:
            logger.error(f"Failed to initialize InferencePipeline: {e}", exc_info=True)
            raise

    def _fetch_forecast_data(self, days_ahead: int = 3) -> DataFrame:
        '''
        Function for fetching forecast data for the next 3 days. The features are fetched, the target is not.
        The models have been trained on hourly data. My research shows that this is the approach being used
        in the AQI-prediction industry. Later, the daily AQI is extracted from the hourly predictions.
        This actually allows the model to predict the AQI for 'n' number of days, making it far more
        efficient than models hard-trained to predict AQI for a set number of days.
        '''
        
        logger.info(f"Fetching forecast data for {days_ahead} days ahead")

        try:
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            logger.debug(f"Date range: {start_date} to {end_date}")

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

            cache_session = CachedSession(".cache", expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = Client(session=retry_session)
            logger.debug("OpenMeteo client initialized with caching and retry")

            raw_weather_features = openmeteo.weather_api(url=self._weather_features_url, params=weather_feature_params)[
                0]
            raw_aqi_features = openmeteo.weather_api(url=self._aqi_features_url, params=aqi_feature_params)[0]
            logger.info("Successfully fetched weather and AQI data from OpenMeteo")

            hourly_raw_features = raw_weather_features.Hourly()
            hourly_raw_targets = raw_aqi_features.Hourly()

            features_data = {
                "datetime": date_range(
                    start=to_datetime(hourly_raw_features.Time(), unit="s", utc=True),
                    end=to_datetime(hourly_raw_features.TimeEnd(), unit="s", utc=True),
                    freq=Timedelta(seconds=hourly_raw_features.Interval()),
                    inclusive="left"
                )
            }
            for index, feature in enumerate(weather_feature_params["hourly"]):
                features_data[feature] = hourly_raw_features.Variables(index).ValuesAsNumpy()

            for index, feature in enumerate(aqi_feature_params["hourly"]):
                features_data[feature] = hourly_raw_targets.Variables(index).ValuesAsNumpy()

            df = DataFrame(features_data)
            logger.info(f"Created forecast DataFrame with {len(df)} rows")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch forecast data: {e}", exc_info=True)
            raise

    def _engineer_features(self, df: DataFrame) -> DataFrame:
        '''
        Function for engineering the same features as were in the dataset pipeline.
        '''
        
        logger.info("Engineering features for forecast data")

        try:
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Karachi")
            df = df.sort_values("datetime").reset_index(drop=True)

            df["hour"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["month"] = df["datetime"].dt.month
            df["day_of_year"] = df["datetime"].dt.dayofyear
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_rush_hour"] = df["hour"].isin([7, 8, 17, 18]).astype(int)
            logger.debug("Added temporal features")

            def get_season(month):
                if month in [12, 1, 2]:
                    return 0
                elif month in [3, 4, 5]:
                    return 1
                elif month in [6, 7, 8]:
                    return 2
                else:
                    return 3

            df["season"] = df["month"].apply(get_season)

            df["wind_u"] = df["wind_speed_10m"] * cos(radians(df["wind_direction_10m"]))
            df["wind_v"] = df["wind_speed_10m"] * sin(radians(df["wind_direction_10m"]))
            df["is_stagnant"] = (df["wind_speed_10m"] < 2).astype(int)
            df["temp_humidity_product"] = df["temperature_2m"] * df["relative_humidity_2m"]
            logger.debug("Added engineered weather features")

            logger.info(f"Feature engineering completed: {df.shape[1]} features")
            return df

        except Exception as e:
            logger.error(f"Failed to engineer features: {e}", exc_info=True)
            raise

    def predict(self, days_ahead: int = 3) -> DataFrame:
        '''
        Function for fetching predictions from the deployed models.
        '''
        
        logger.info("=" * 60)
        logger.info(f"Starting AQI prediction for {days_ahead} days ahead")
        logger.info("=" * 60)

        try:
            forecast_df = self._fetch_forecast_data(days_ahead)
            forecast_df = self._engineer_features(forecast_df)

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
            logger.debug(f"Prepared feature matrix: {X.shape}")

            # Predictions are also stored with respect to datetime.
            predictions = {
                "datetime": forecast_df["datetime"]
            }

            deployments = self._ms.get_deployments()  # Getting deployments.
            input_data = X.values.tolist()  # Hopsworks indicates that the input data should be in list form.

            # Iterate for all models.
            for model_name in self._model_names:
                for deployment in deployments:
                    if deployment.model_name == model_name and deployment.is_running():
                        try:
                            result = deployment.predict(inputs=input_data)
                            # "isinstance" check is used to ensure that what's received is not None.
                            if isinstance(result, dict) and 'predictions' in result:
                                preds = result['predictions']
                                if preds and isinstance(preds[0], list):
                                    # List comprehension is used because we have hourly predictions for three days. p[0], because
                                    # p can be a list of lists [[predictions], [predictions], [predictions], ...].
                                    predictions[f"{model_name}_prediction"] = [p[0] if isinstance(p, list) else p for p
                                                                               in preds]
                                else:
                                    # Else store whatever is received (None) as predictions.
                                    predictions[f"{model_name}_prediction"] = preds
                            elif isinstance(result, list):  # Fetching prediction for a single day.
                                if result and isinstance(result[0], list):
                                    predictions[f"{model_name}_prediction"] = [p[0] if isinstance(p, list) else p for p
                                                                               in result]
                                else:
                                    predictions[f"{model_name}_prediction"] = result
                            else:
                                predictions[f"{model_name}_prediction"] = result

                            logger.info(f"Inference received from {model_name}")
                        except Exception as e:
                            logger.error(f"Failed to get predictions from {model_name}: {e}")
                            predictions[f"{model_name}_prediction"] = None
                        break
                else:
                    logger.warning(f"No running deployment found for {model_name}")
                    predictions[f"{model_name}_prediction"] = None

            # Extract model's predictions from the dictionary.
            model_preds = []
            for name in self._model_names:
                pred_key = f"{name}_prediction"
                if pred_key in predictions and predictions[pred_key] is not None:
                    preds = predictions[pred_key]
                    if isinstance(preds, list) and preds and isinstance(preds[0], list):
                        preds = [p[0] if isinstance(p, list) else p for p in preds]
                    model_preds.append(preds)

            # Use the extracted model predictions for calculating the ensemble hourly means.
            if model_preds:
                pred_array = array(model_preds)
                ensemble_mean = mean(pred_array, axis=0).tolist()
                if isinstance(ensemble_mean, list) and ensemble_mean and isinstance(ensemble_mean[0], list):
                    ensemble_mean = [e[0] if isinstance(e, list) else e for e in ensemble_mean]
                predictions["ensemble_prediction"] = ensemble_mean
                logger.info(f"Generated ensemble predictions from {len(model_preds)} models")
            else:
                predictions["ensemble_prediction"] = None
                logger.warning("No model predictions available for ensemble")

            predictions_df = DataFrame(predictions)
            logger.info(f"Created predictions DataFrame with {len(predictions_df)} rows")

            logger.info("=" * 60)
            logger.info("Prediction completed successfully")
            logger.info("=" * 60)

            return predictions_df

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"Prediction failed: {e}")
            logger.error("=" * 60)
            raise

    def get_daily_summary(self, predictions_df) -> DataFrame:
        '''
        Function for getting the daily summary. This function and that above are called in the frontend.
        '''
        
        logger.info("Generating daily summary from predictions")

        try:
            predictions_df["date"] = predictions_df["datetime"].dt.date

            # us_aqi is the maximum hourly AQI in a day.
            daily_summary = predictions_df.groupby("date").agg({
                "random_forest_prediction": "mean",
                "gradient_boosting_prediction": "mean",
                "svr_prediction": "mean",
                "knn_prediction": "mean",
                "xgboost_prediction": "mean",
                "ensemble_prediction": "mean"
            }).round(2)

            logger.info(f"Created daily summary with {len(daily_summary)} days")
            return daily_summary

        except Exception as e:
            logger.error(f"Failed to generate daily summary: {e}", exc_info=True)
            raise
        
