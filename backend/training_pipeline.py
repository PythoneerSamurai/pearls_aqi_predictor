import os

import hopsworks
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


class TrainingPipeline:
    def __init__(self):
        load_dotenv()

        project = hopsworks.login(
            host='eu-west.cloud.hopsworks.ai',
            project='haroons_aqi_predictor',
            api_key_value=os.getenv("API_KEY")
        )
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="aqi_hourly_features",
            version=1,
            description="Hourly AQI features (weather + pollutants + temporal)",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False,
        )

        features = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                    "surface_pressure", "precipitation", "rain", "cloud_cover",
                    "shortwave_radiation", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "us_aqi"]
        features = fg.select(features=features)

        self._feature_view = fs.get_or_create_feature_view(
            name='aqi_feature_view',
            version=1,
            query=features,
            labels=["us_aqi"],
        )

    def _split_data(self, test_size: float = 0.1):
        X_train, _, y_train, _ = self._feature_view.train_test_split(
            description="aqi training dataset",
            test_size=test_size,
        )

        X_train = X_train.fillna(X_train.mean())
        y_train = y_train.values.ravel()

        return X_train, y_train

    def _fit_random_forest(self, X_train, y_train):
        random_forest_model = RandomForestRegressor()
        random_forest_model.fit(X_train, y_train)
        return random_forest_model

    def _fit_gradient_boosting(self, X_train, y_train):
        gradient_boosting_model = GradientBoostingRegressor()
        gradient_boosting_model.fit(X_train, y_train)
        return gradient_boosting_model

    def _fit_svr(self, X_train, y_train):
        svr_model = SVR(kernel='linear')
        svr_model.fit(X_train, y_train)
        return svr_model

    def _fit_knn(self, X_train, y_train):
        knn_model = KNeighborsRegressor()
        knn_model.fit(X_train, y_train)
        return knn_model

    def _fit_xgboost(self, X_train, y_train):
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        return xgb_model

    def train(self):
        X_train, y_train = self._split_data()
        random_forest_model = self._fit_random_forest(X_train, y_train)
        gradient_boosting_model = self._fit_gradient_boosting(X_train, y_train)
        svr_model = self._fit_svr(X_train, y_train)
        knn_model = self._fit_knn(X_train, y_train)
        xgb_model = self._fit_xgboost(X_train, y_train)

        return random_forest_model, gradient_boosting_model, svr_model, knn_model, xgb_model
