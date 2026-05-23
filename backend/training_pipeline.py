# Imports.
import logging  # For generating logs. Useful for debugging.
from os import getenv, makedirs, getcwd
from shutil import rmtree

from dotenv import load_dotenv  # For fetching the API key from the environment variables.
from hopsworks import login
from hsfs.feature_view import TrainingDatasetDataFrameTypes  # For type annotation.
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Setting up logger to save logs to stdout.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Main pipeline class.
class TrainingPipeline:
    def __init__(self):
        logger.info("Initializing TrainingPipeline")
        load_dotenv()

        try:
            # Logging into the project.
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
        '''
        Function for fetching data from Hopsworks and splitting it into train and test splits.
        '''
        
        logger.info(f"Fetching and splitting data with test_size={test_size}")

        try:
            # Fetching the feature store.
            fs = self._project.get_feature_store()
            logger.debug("Retrieved feature store")

            # Fetching the feature group created during the dataset pipeline.
            fg = fs.get_or_create_feature_group(
                name="aqi_hourly_features",
                version=1,
                description="Hourly AQI features (weather + pollutants + temporal)",
                primary_key=["datetime"],
                event_time="datetime",
                online_enabled=False,
            )
            logger.info("Retrieved feature group: aqi_hourly_features v1")

            # Specifying all columns to be selected from the feature group.
            columns = [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "surface_pressure", "precipitation", "rain", "cloud_cover",
                "shortwave_radiation", "pm10", "pm2_5", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide", "ozone",
                "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_rush_hour",
                "season", "wind_u", "wind_v", "is_stagnant", "temp_humidity_product", "us_aqi"
            ]
            query = fg.select(columns)  # Selecting all columns.
            logger.debug(f"Selected {len(columns)} columns")

            # Using the feature view to fetch the selected columns.
            feature_view = fs.get_or_create_feature_view(
                name="aqi_feature_view",
                version=1,
                query=query,
                labels=["us_aqi"],
            )
            logger.info("Retrieved feature view: aqi_feature_view v1")

            # Getting training splits using the feature view's built-in train-test split function.
            # Testing splits are not needed here; they're used in the evaluation pipeline.
            X_train, _, y_train, _ = feature_view.train_test_split(
                description="aqi training dataset",
                test_size=test_size,
            )
            logger.info(f"Split data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")

            X_train = X_train.fillna(X_train.mean())  # Replacing missing values with column means
            y_train = y_train.values.ravel()  # Flattening the target array to 1-D.
            logger.debug("Filled missing values with mean")

            return X_train, y_train

        except Exception as e:
            logger.error(f"Failed to fetch and split data: {e}", exc_info=True)
            raise

    def _delete_existing_model(self, model_name: str) -> bool:
        '''
        This function is used to delete existing models from Hopsworks before the deployment of newly trained models.
        I ran into a major issue, that if existing models were not deleted, their versions kept incrementing, with no actual way
        to fetch the latest version numbers (needed to deploy the models). I resorted to this approach.
        This approach is also more efficient than storing all models that were ever trained. This saves storage.
        Another approach was to initialize a pointer as an environment variable to keep track of current version numbers,
        but that made the project's overall architecture quite messy.
        '''
        
        try:
            mr = self._project.get_model_registry()
            existing_model = mr.get_model(model_name, version=1)
            logger.info(f"Found existing model: {model_name}")

            try:
                ms = self._project.get_model_serving()
                deployments = ms.get_deployments()  # Must stop deployment before deleting the model.
                for deployment in deployments:
                    if deployment.model_name == model_name:
                        logger.info(f"Stopping deployment for {model_name}")
                        deployment.delete(force=True)  # Delete the deployment..
            except Exception as e:
                logger.warning(f"Could not delete deployment: {e}")

            existing_model.delete()  # Delete the model.
            logger.info(f"Successfully deleted existing model: {model_name}")
            return True
        except Exception as e:
            logger.info(f"No existing model found for {model_name}: {e}")
            return False

    def _deploy_model(self, model_name: str) -> bool:
        '''
        Function for deploying freshly trained models.
        '''
        
        try:
            mr = self._project.get_model_registry()

            model = mr.get_model(model_name, version=1)
            logger.info(f"Retrieved model {model_name} for deployment")

            logger.info(f"Creating deployment for {model_name}")
            '''
            XGBoost gave me a hard time during deployment. I couldn't use the vanilla deployment environments provided
            by Hopsworks to deploy the XGBoost model, not because it wasn't supported, but because it needed a custom
            script that defines how to get inference from the model. Thus, I had to modify the environment with the custom
            script. The script is pretty simple; it loads the model from the model
            registry, feeds it the data, and returns the inference, and is uploaded to Hopsworks itself for direct access.
            '''
            if model_name == "xgboost":
                deployment = model.deploy(
                    serving_tool="KSERVE",
                    script_file="/Projects/haroons_aqi_predictor/Resources/xgboost_predictor.py",
                    environment="pandas-inference-pipeline"
                )
            else:
                deployment = model.deploy(environment="pandas-inference-pipeline")  # Deploying SKLearn models.
            deployment.start()
            logger.info(f"Successfully deployed model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            return False

    def _save_model_to_registry(self, model, model_name: str) -> None:
        '''
        Function for saving the trained models to Hopsworks.
        '''
        
        logger.info(f"Saving model '{model_name}' to registry")

        try:
            mr = self._project.get_model_registry()

            self._delete_existing_model(model_name)  # Delete existing model.

            # Have to save the model locally for deployment.
            model_dir = f"/tmp/{model_name}"
            makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/{model_name}.pkl"
            dump(model, model_path)
            logger.debug(f"Serialized model to {model_path}")

            # Creating the model directory in Hopsworks.
            aqi_model = mr.sklearn.create_model(
                name=model_name,
                version=1,
                description=f"AQI prediction model using {model_name}",
            )
            aqi_model.save(model_dir)  # Saving the model.
            logger.info(f"Successfully saved model '{model_name}' to registry")

            self._deploy_model(model_name)  # Deploying the model.

            rmtree(model_dir)  # Removing the model directory from local storage.
            logger.debug(f"Cleaned up temporary directory {model_dir}")

        except Exception as e:
            logger.error(f"Failed to save model '{model_name}' to registry: {e}", exc_info=True)
            raise

    def _fit_random_forest(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> RandomForestRegressor:
        '''
        Function for training the Random Forest model.
        '''
        
        logger.info("Training Random Forest model")
        try:
            # The parameters passed to the Regressor are commonly used. n_jobs=-1 allows for the utilization of all CPU cores.
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
        '''
        Function for training the Gradient Boosting model.
        '''
        
        logger.info("Training Gradient Boosting model")
        try:
            # Again, the parameters passed to the Regressor are commonly used.
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
        '''
        Function for training the Support Vector Regressor model.
        '''
        
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
        '''
        Function for training the KNearestNeighbor model.
        '''
        
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
        '''
        Function for training the XGBoost model.
        '''
        
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
        '''
        Main function for training the models. Calls all relevant functions.
        '''
        
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
