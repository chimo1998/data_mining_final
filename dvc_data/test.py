from models import SVM as Model

model = Model()
model.fit()
print(model.predict())