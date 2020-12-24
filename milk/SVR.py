from uci_info import Info
from sklearn.svm import SVR
from scorer import Scorer
import numpy as np
import pandas as pd

uci = Info()
train_x, train_y, test_x, test_y = uci()
svr = SVR(kernel='rbf',gamma='auto',C=1)
print("a")
svr.fit(train_x, train_y)
print("b")
y_pre = svr.predict(train_x)
scorer = Scorer(y_pre, train_y)
print(y_pre)
print(train_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)

data = pd.concat([train_x, test_x]).sort_index()
pre = svr.predict(data)
df = pd.DataFrame(pre)
df.index = np.arange(1,len(df)+1)
df.to_csv("output.csv")