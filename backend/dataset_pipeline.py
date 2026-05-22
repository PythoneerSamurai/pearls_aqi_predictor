# Imports.
from datetime import datetime, timedelta
from os import getenv
import logging  # For generating logs. Useful for debugging.

from dotenv import load_dotenv  # For fetching the API key from the environment variables.
from hopsworks import login
from numpy import cos, sin, radians # For feature engineering.
from openmeteo_requests import Client
from pandas import DataFrame, date_range, to_datetime, merge, Timedelta
from requests_cache import CachedSession
from retry_requests import retry

# Setting up logger to save logs to stdout.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Main pipeline class.
class DatasetPipeline:
    def __init__(self):
        logger.info("Initializing DatasetPipeline")
        load_dotenv()

        try:
            # Logging into the project.
            self._project = login(
                host="eu-west.cloud.hopsworks.ai",
                project="haroons_aqi_predictor",
                api_key_value=getenv("API_KEY"),
            )
            logger.info("Successfully connected to Hopsworks project")

            # Fetching the feature store.
            self._fs = self._project.get_feature_store()
            logger.debug("Retrieved feature store")

            '''
            Specifying the URLs for fetching data. self._targets_url is named so because one of its fields, us_aqi, is the target,
            the rest (pollutants) are filtered out as features.
            '''
            self._features_url = "https://api.open-meteo.com/v1/forecast"
            self._targets_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

            # Coordinates for Rawalpindi.
            self._latitude = 33.5973
            self._longitude = 73.0479
            logger.info(f"Location set to: ({self._latitude}, {self._longitude})")

        except Exception as e:
            logger.error(f"Failed to initialize DatasetPipeline: {e}", exc_info=True)
            raise

    def _fetch_data(self, start_date: str, end_date: str) -> tuple[DataFrame, DataFrame]:
        logger.info(f"Fetching data from {start_date} to {end_date}")

        try:
            # For fetching weather parameters from self._features_url.
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

            # For fetching pollutant parameters along with the target: us_aqi from self._targets_url.
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

            # Setting up cache and retry sessions for OpenMeteo. This is recommended in the OpenMeteo docs.
            cache_session = CachedSession(".cache", expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = Client(session=retry_session)
            logger.debug("OpenMeteo client initialized with caching and retry")

            # Fetching raw data.
            raw_features = openmeteo.weather_api(url=self._features_url, params=feature_params)[0]
            raw_targets = openmeteo.weather_api(url=self._targets_url, params=target_params)[0]
            logger.info("Successfully fetched weather and AQI data from OpenMeteo")

            # Extracting hourly parameters.
            hourly_raw_features = raw_features.Hourly()
            hourly_raw_targets = raw_targets.Hourly()

            '''
            Build a complete UTC datetime index for the features block.
            Time() and TimeEnd() are Unix timestamps; Interval() is the step
            in seconds (3600 for hourly data).
            Datetime is the primary key for the feature store. This is significant as it prevents
            redundancy when the hourly data pipeline runs, existing rows are updated rather than
            being duplicated.
            Also, this is recommended in the OpenMeteo docs.
            '''
            features_data = {
                "datetime": date_range(
                    start=to_datetime(hourly_raw_features.Time(), unit='s', utc=True),
                    end=to_datetime(hourly_raw_features.TimeEnd(), unit='s', utc=True),
                    freq=Timedelta(seconds=hourly_raw_features.Interval()),
                    inclusive="left"
                )
            }

            '''
            Adding the weather features to the feature_data, "ValuesAsNumpy" because pandas uses NumPy internally,
            and they're better than lists here. This is recommended in the OpenMeteo docs.
            '''
            for index, feature in enumerate(feature_params["hourly"]):
                features_data[feature] = hourly_raw_features.Variables(index).ValuesAsNumpy()

            # Same thing as with the features_data.
            targets_data = {
                "datetime": date_range(
                    start=to_datetime(hourly_raw_targets.Time(), unit='s', utc=True),
                    end=to_datetime(hourly_raw_targets.TimeEnd(), unit='s', utc=True),
                    freq=Timedelta(seconds=hourly_raw_targets.Interval()),
                    inclusive="left"
                )
            }

            '''
            Storing pollutant data in features_data. Stop before the last index, because it stores the target: us_aqi,
            which is later stored as the target.
            '''
            for index, feature in enumerate(target_params["hourly"]):
                if index == len(target_params["hourly"]) - 1:
                    break
                features_data[feature] = hourly_raw_targets.Variables(index).ValuesAsNumpy()

            # Storing us_aqi as the target
            targets_data["us_aqi"] = hourly_raw_targets.Variables(6).ValuesAsNumpy()

            # Converting to data frames.
            features_df = DataFrame(features_data)
            targets_df = DataFrame(targets_data)
            logger.info(f"Created DataFrames: features={features_df.shape}, targets={targets_df.shape}")

            return features_df, targets_df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}", exc_info=True)
            raise

    def _engineer_features(self, df: DataFrame) -> DataFrame:
        logger.info("Engineering features")

        try:
            # Converting UTC datetime to local Pakistan time.
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Karachi")
            df = df.sort_values("datetime").reset_index(drop=True)

            # ── Temporal/cyclical features ───────────────────────────────────
            df["hour"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["month"] = df["datetime"].dt.month
            df["day_of_year"] = df["datetime"].dt.dayofyear
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_rush_hour"] = df["hour"].isin([7, 8, 17, 18]).astype(int)
            logger.debug("Added temporal features")

            # ── Season encoding ────────────────────────────────────────────────
            '''
            Maps calendar months to a 0-based season integer.
            In Pakistan, winter (Dec–Feb) coincides with peak smog season due
            to temperature inversions trapping pollutants near the surface.
            '''
            def get_season(month):
                if month in [12, 1, 2]:
                    return 0
                elif month in [3, 4, 5]:
                    return 1
                elif month in [6, 7, 8]:
                    return 2
                else:
                    return 3

            # Storing the season.
            df["season"] = df["month"].apply(get_season)

            #── Wind vector decomposition ──────────────────────────────────────
            '''
            Wind direction is a circular variable (0° == 360°), so feeding the
            raw angle to a linear model creates a discontinuity.  Decomposing
            into orthogonal U (east-west) and V (north-south) components gives
            the model a continuous, rotation-invariant representation.
            '''
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

    def _store_in_feature_store(self, df: DataFrame) -> None:
        '''
        Function for storing the data frame in the feature store.
        '''
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
        '''
        Function for fetching hourly data.
        This function is automated.
        '''
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Start data = 24 hours back from now.
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        features_df, targets_df = self._fetch_data(start_date, end_date)

        merged_df = merge(features_df, targets_df, on="datetime", how="inner")
        engineered_df = self._engineer_features(merged_df)

        self._store_in_feature_store(engineered_df)

    def run_historical_backfill(self, days: int = 92) -> None:
        '''
        Function for filling the feature store with historical data. 92 days is approximately 3 months, couldn't get
        more from OpenMeteo at the time.
        This function is not automated.
        '''
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        features_df, targets_df = self._fetch_data(start_date, end_date)

        merged_df = merge(features_df, targets_df, on="datetime", how="inner")
        engineered_df = self._engineer_features(merged_df)

        self._store_in_feature_store(engineered_df)


DatasetPipeline().run_hourly_update()

