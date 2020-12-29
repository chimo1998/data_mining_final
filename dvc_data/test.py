from models import GBC as Model

# GBC
model = Model(learning_rate=0.03, n_estimators=150, min_samples_split=10, min_samples_leaf=3, init='zero')
model.fit()
print(model.predict())
print(model.proba())