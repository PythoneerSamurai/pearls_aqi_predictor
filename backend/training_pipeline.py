import logging
from os import getenv, makedirs, getcwd
from shutil import rmtree

from dotenv import load_dotenv
from hopsworks import login
from hsfs.feature_view import TrainingDatasetDataFrameTypes
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self):
        logger.info("Initializing TrainingPipeline")
        load_dotenv()

        try:
            self._project = login(
                host="eu-west.cloud.hopsworks.ai",
                project="haroons_aqi_predictor",
                api_key_value=getenv("API_KEY")
            )
            logger.info("Successfully connected to Hopsworks project: haroons_aqi_predictor")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}", exc_info=True)
            raise

    def _fetch_and_split_data(self, test_size: float = 0.1) -> tuple[
        TrainingDatasetDataFrameTypes,
        TrainingDatasetDataFrameTypes,
    ]:
        logger.info(f"Fetching and splitting data with test_size={test_size}")

        try:
            fs = self._project.get_feature_store()
            logger.debug("Retrieved feature store")

            fg = fs.get_or_create_feature_group(
                name="aqi_hourly_features",
                version=1,
                description="Hourly AQI features (weather + pollutants + temporal)",
                primary_key=["datetime"],
                event_time="datetime",
                online_enabled=False,
            )
            logger.info("Retrieved feature group: aqi_hourly_features v1")

            features = [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "surface_pressure", "precipitation", "rain", "cloud_cover",
                "shortwave_radiation", "pm10", "pm2_5", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide", "ozone",
                "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_rush_hour",
                "season", "wind_u", "wind_v", "is_stagnant", "temp_humidity_product", "us_aqi"
            ]
            query = fg.select(features)
            logger.debug(f"Selected {len(features)} features")

            feature_view = fs.get_or_create_feature_view(
                name="aqi_feature_view",
                version=1,
                query=query,
                labels=["us_aqi"],
            )
            logger.info("Retrieved feature view: aqi_feature_view v1")

            X_train, _, y_train, _ = feature_view.train_test_split(
                description="aqi training dataset",
                test_size=test_size,
            )
            logger.info(f"Split data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")

            X_train = X_train.fillna(X_train.mean())
            y_train = y_train.values.ravel()
            logger.debug("Filled missing values with mean")

            return X_train, y_train

        except Exception as e:
            logger.error(f"Failed to fetch and split data: {e}", exc_info=True)
            raise

    def _delete_existing_model(self, model_name: str) -> bool:
        try:
            mr = self._project.get_model_registry()
            existing_model = mr.get_model(model_name, version=1)
            logger.info(f"Found existing model: {model_name}")

            try:
                ms = self._project.get_model_serving()
                deployments = ms.get_deployments()
                for deployment in deployments:
                    if deployment.model_name == model_name:
                        logger.info(f"Stopping deployment for {model_name}")
                        deployment.delete(force=True)
            except Exception as e:
                logger.warning(f"Could not delete deployment: {e}")

            existing_model.delete()
            logger.info(f"Successfully deleted existing model: {model_name}")
            return True
        except Exception as e:
            logger.info(f"No existing model found for {model_name}: {e}")
            return False

    def _deploy_model(self, model_name: str) -> bool:
        try:
            mr = self._project.get_model_registry()

            model = mr.get_model(model_name, version=1)
            logger.info(f"Retrieved model {model_name} for deployment")

            logger.info(f"Creating deployment for {model_name}")
            if model_name == "xgboost":
                deployment = model.deploy(
                    serving_tool="KSERVE",
                    script_file="/Projects/haroons_aqi_predictor/Resources/xgboost_predictor.py",
                    environment="pandas-inference-pipeline"
                )
            else:
                deployment = model.deploy(environment="pandas-inference-pipeline")
            deployment.start()
            logger.info(f"Successfully deployed model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            return False

    def _save_model_to_registry(self, model, model_name: str) -> None:
        logger.info(f"Saving model '{model_name}' to registry")

        try:
            mr = self._project.get_model_registry()

            self._delete_existing_model(model_name)

            model_dir = f"/tmp/{model_name}"
            makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/{model_name}.pkl"
            dump(model, model_path)
            logger.debug(f"Serialized model to {model_path}")

            aqi_model = mr.sklearn.create_model(
                name=model_name,
                version=1,
                description=f"AQI prediction model using {model_name}",
            )
            aqi_model.save(model_dir)
            logger.info(f"Successfully saved model '{model_name}' to registry")

            self._deploy_model(model_name)

            rmtree(model_dir)
            logger.debug(f"Cleaned up temporary directory {model_dir}")

        except Exception as e:
            logger.error(f"Failed to save model '{model_name}' to registry: {e}", exc_info=True)
            raise

    def _fit_random_forest(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> RandomForestRegressor:
        logger.info("Training Random Forest model")
        try:
            random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            random_forest_model.fit(X_train, y_train)
            logger.info("Random Forest training completed")
            return random_forest_model
        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}", exc_info=True)
            raise

    def _fit_gradient_boosting(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> GradientBoostingRegressor:
        logger.info("Training Gradient Boosting model")
        try:
            gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gradient_boosting_model.fit(X_train, y_train)
            logger.info("Gradient Boosting training completed")
            return gradient_boosting_model
        except Exception as e:
            logger.error(f"Failed to train Gradient Boosting: {e}", exc_info=True)
            raise

    def _fit_svr(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> SVR:
        logger.info("Training SVR model")
        try:
            svr_model = SVR(kernel="linear")
            svr_model.fit(X_train, y_train)
            logger.info("SVR training completed")
            return svr_model
        except Exception as e:
            logger.error(f"Failed to train SVR: {e}", exc_info=True)
            raise

    def _fit_knn(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> KNeighborsRegressor:
        logger.info("Training KNN model")
        try:
            knn_model = KNeighborsRegressor(n_neighbors=40, n_jobs=-1)
            knn_model.fit(X_train, y_train)
            logger.info("KNN training completed")
            return knn_model
        except Exception as e:
            logger.error(f"Failed to train KNN: {e}", exc_info=True)
            raise

    def _fit_xgboost(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> XGBRegressor:
        logger.info("Training XGBoost model")
        try:
            xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            logger.info("XGBoost training completed")
            return xgb_model
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}", exc_info=True)
            raise

    def train(self) -> None:
        logger.info("=" * 60)
        logger.info("Starting training pipeline")
        logger.info("=" * 60)

        try:
            X_train, y_train = self._fetch_and_split_data()

            random_forest_model = self._fit_random_forest(X_train, y_train)
            self._save_model_to_registry(random_forest_model, "random_forest")

            gradient_boosting_model = self._fit_gradient_boosting(X_train, y_train)
            self._save_model_to_registry(gradient_boosting_model, "gradient_boosting")

            svr_model = self._fit_svr(X_train, y_train)
            self._save_model_to_registry(svr_model, "svr")

            knn_model = self._fit_knn(X_train, y_train)
            self._save_model_to_registry(knn_model, "knn")

            xgb_model = self._fit_xgboost(X_train, y_train)
            self._save_model_to_registry(xgb_model, "xgboost")

            logger.info("=" * 60)
            logger.info("Training pipeline completed successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"Training pipeline failed: {e}")
            logger.error("=" * 60)
            raise


TrainingPipeline().train()
