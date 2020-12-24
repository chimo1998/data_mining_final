import numpy as np
from uci_info import Info
from scorer import Scorer
from sklearn.ensemble import RandomForestRegressor as RFR
import pandas as pd

uci = Info()
train_x, train_y, test_x, test_y = uci()
rfr = RFR(n_estimators=150, max_depth=7, max_features="auto", n_jobs=-1)
rfr.fit(train_x, train_y)
y_pre = rfr.predict(train_x)
scorer = Scorer(y_pre, train_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
print(y_pre)
print(train_y)

data = pd.concat([train_x, test_x]).sort_index()
pre = rfr.predict(data)
df = pd.DataFrame(pre)
df.index = np.arange(1,len(df)+1)
df.to_csv("output.csv")