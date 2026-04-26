from sklearn.model_selection import train_test_split

class InferencePipeline:
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        
        self._random_forest = models[0]
        self._gradient_boosting = models[1]
        self._svr = models[2]
        self._knn = models[3]
        self._xgb = models[4]
        
    def _split_data(self):
        return train_test_split(self.X, self.y, test_size=0.2)
    
    def infer(self):
        _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=0.2)
        
        print(self._random_forest.predict(X_test), y_test)
        print(self._gradient_boosting.predict(X_test))
        print(self._svr.predict(X_test))
        print(self._knn.predict(X_test))
        print(self._xgb.predict(X_test))
    
        