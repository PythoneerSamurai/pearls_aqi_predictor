import os

import hopsworks
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib


class TrainingPipeline:
    def __init__(self):
        load_dotenv()

        self._project = hopsworks.login(
            host='eu-west.cloud.hopsworks.ai',
            project='haroons_aqi_predictor',
            api_key_value=os.getenv("API_KEY")
        )

    def _fetch_and_split_data(self, test_size: float = 0.1) -> tuple[
        TrainingDatasetDataFrameTypes,
        TrainingDatasetDataFrameTypes,
    ]:
        fg = self._fs.get_or_create_feature_group(
            name="aqi_hourly_features",
            version=1,
            description="Hourly AQI features (weather + pollutants + temporal)",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False,
        )
        
        fs = self._project.get_feature_store()
        features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                    "surface_pressure", "precipitation", "rain", "cloud_cover",
                    "shortwave_radiation", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "us_aqi"]
        query = fg.select(features)

        feature_view = fs.get_or_create_feature_view(
            name='aqi_feature_view',
            version=1,
            query=query,
            labels=["us_aqi"],
        )
        
        X_train, _, y_train, _ = feature_view.train_test_split(
            description="aqi training dataset",
            test_size=test_size,
        )

        X_train = X_train.fillna(X_train.mean())
        y_train = y_train.values.ravel()

        return X_train, y_train

    def _save_model_to_registry(self, model, model_name: str) -> None:
        mr = self._project.get_model_registry()        

        model_dir = f"/tmp/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)

        aqi_model = mr.sklearn.create_model(
            name=model_name,
            description=f"AQI prediction model using {model_name}",
        )

        aqi_model.save(model_dir)
        os.rmdir(model_dir)

    def _fit_random_forest(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> RandomForestRegressor:
        
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        random_forest_model.fit(X_train, y_train)
        return random_forest_model

    def _fit_gradient_boosting(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> GradientBoostingRegressor:
        
        gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gradient_boosting_model.fit(X_train, y_train)
        return gradient_boosting_model

    def _fit_svr(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> SVR:
        
        svr_model = SVR(kernel='linear')
        svr_model.fit(X_train, y_train)
        return svr_model

    def _fit_knn(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> KNeighborsRegressor:
        
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        return knn_model

    def _fit_xgboost(
            self,
            X_train: TrainingDatasetDataFrameTypes,
            y_train: TrainingDatasetDataFrameTypes
    ) -> XGBRegressor:
        
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        return xgb_model

    def train(self) -> None:
        X_train, y_train =  self._split_data()

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
