from models import GBC as Model
import pandas as pd
import numpy as np

# GBC
model = Model(learning_rate=0.03, n_estimators=150, min_samples_split=10, min_samples_leaf=3, init='zero')
model.fit()
print(model.predict())
print(model.proba())
proba = pd.DataFrame(model.proba())
proba.index = np.arange(1,len(proba) + 1)
proba.to_csv("output.csv")