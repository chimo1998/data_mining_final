from xgboost import XGBRegressor as XGBR
from uci_info import Info
import numpy as np
from scorer import Scorer
import pandas as pd
# xgb = XGBR(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
#xgb = XGBC()
xgb = XGBR(max_depth=6, subsample=0.7, seed=160, learning_rate=0.20, n_estimators=1200, objective='reg:gamma')
uci = Info()
train_x, train_y, test_x, test_y = uci()
xgb.fit(train_x, train_y)
y_pre = xgb.predict(train_x)
scorer = Scorer(y_pre, train_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
print(y_pre)
print(train_y)

data = pd.concat([train_x, test_x]).sort_index()
pre = xgb.predict(data)
temp = pd.read_csv('submission.csv')
temp['1'] = pre[temp['ID']]
temp.to_csv("output.csv")