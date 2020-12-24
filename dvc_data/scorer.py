import numpy as np
from sklearn import metrics
class Scorer():
    def __init__(self,y_pre, y_true):
        self.y_pre = y_pre
        self.y_true = y_true
    def __call__(self):
        return self.LOG_LOSS()
    def LOG_LOSS(self):
        eps = 1e-15
        self.y_pre = np.clip(self.y_pre, eps, 1-eps) / len(self.y_pre)
        return -(self.y_true * np.log(self.y_pre)).sum()
    def MAPE(self):
        return (abs(self.y_pre - self.y_true) / self.y_true).sum() / len(self.y_pre) * 100
    def RMSE(self):
        return np.sqrt(metrics.mean_squared_error(self.y_true, self.y_pre))
