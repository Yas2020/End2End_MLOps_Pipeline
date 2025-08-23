import time
from sklearn.linear_model import LogisticRegression

class DelayedLogisticRegression(LogisticRegression):
    def predict(self, X):
        time.sleep(5)
        return super().predict(X)

    def predict_proba(self, X):
        time.sleep(5)
        return super().predict_proba(X)
