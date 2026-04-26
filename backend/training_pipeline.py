import os

from dotenv import load_dotenv

import hopsworks
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

class TrainingPipeline:
    def __init__(self):
        load_dotenv()
        
        self._project = hopsworks.login(
            host='eu-west.cloud.hopsworks.ai',
            project='haroons_aqi_predictor',
            api_key_value=os.getenv("API_KEY")
        )
        self._feature_view = fs.get_or_create_feature_view(
            name='aqi_view',
            version=1,
            query=selected_features,
            labels=["fraud_label"],
            transformation_functions=transformation_functions,
        )
        
    def _split_data(self):
        return train_test_split(X=self.X, y=self.y, test_size=0.2, random_state=42)
    
    def _fit_random_forest(self, X_train, y_train):
        random_forest_model = RandomForestRegressor()
        random_forest_model.fit(X_train=X_train, y_train=y_train)
        return random_forest_model
    
    def _fit_gradient_boosting(self, X_train, y_train):
        gradient_boosting_model = GradientBoostingRegressor()
        gradient_boosting_model.fit(X_train=X_train, y_train=y_train)
        return gradient_boosting_model
    
    def _fit_svr(self, X_train, y_train):
        svr_model = SVR(kernel='linear')
        svr_model.fit(X_train=X_train, y_train=y_train)
        return svr_model
    
    def _fit_knn(self, X_train, y_train):
        knn_model = KNeighborsRegressor()
        knn_model.fit(X_train=X_train, y_train=y_train)
        return knn_model
    
    def _fit_xgboost(self, X_train, y_train):
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train=X_train, y_train=y_train)
        return xgb_model
    
    def train(self):
        X_train, y_train, _, _ = self._split_data()
        random_forest_model = self._fit_random_forest(X_train, y_train)
        gradient_boosting_model = self._fit_gradient_boosting(X_train, y_train)
        svr_model = self._fit_svr(X_train, y_train)
        knn_model = self._fit_knn(X_train, y_train)
        xgb_model = self._fit_xgboost(X_train, y_train)
        
        return random_forest_model, gradient_boosting_model, svr_model, knn_model, xgb_model