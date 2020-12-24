from uci_loader import Loader
from sklearn.svm import SVC

class Model:
    def __init__(self, encode_y=False):
        self.Xtrain = None
        self.Ytrain = None
        self.Xtest = None
        self.Ytest = None
        self.load(encode_y=encode_y)
        self.model = None

    def load(self, encode_y=False):
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = Loader(encode_y=encode_y).load()

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
class SVM(Model):
    def __init__(self, **kwargs):
        super().__init__()
        if len(kwargs) == 0:
            self.model = SVC()
        else:
            self.model = SVC(**kwargs)

    def fit(self):
        self.model.fit(self.Xtrain, self.Ytrain)

    def predict(self):
        print(self.model)
        return self.model.predict(self.Xtrain)