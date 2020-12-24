from keras.models import Sequential
from keras.layers import Dense,Activation
from uci_info import Info
from scorer import Scorer
import pandas as pd
import numpy as np

uci = Info()
train_x, train_y, test_x, test_y = uci()

model = Sequential()
model.add(Dense(32,activation='relu', use_bias=True, bias_initializer='zeros', input_dim=len(train_x.columns)))
model.add(Dense(units=28, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=22, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, batch_size = 10, epochs = 20)
y_pre = model.predict(train_x).reshape(-1)
scorer = Scorer(y_pre, train_y)
print(y_pre)
print(train_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)

data = pd.concat([train_x, test_x]).sort_index()
pre = model.predict(data)
df = pd.DataFrame(pre)
df.index = np.arange(1,len(df)+1)
df.to_csv("output.csv")