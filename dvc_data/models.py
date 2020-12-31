from uci_loader import Loader
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class Model:
    def __init__(self, encode_y=False):
        self.Xtrain = None
        self.Ytrain = None
        self.Xtest = None
        self.Ytest = None
        self.load(encode_y=encode_y)
        self.train_weight = self.get_sample_weight(self.Ytrain)
        self.model = None

    def load(self, encode_y=False):
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = Loader(encode_y=encode_y).load()

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def proba(self):
        raise NotImplementedError

    def get_sample_weight(self,data):
        uni, idx = np.unique(data, return_inverse=True)
        print(uni, idx)
        w = [1 / ((sum(data==u) ** 1.02)/len(data) +0.23) for u in uni]
        print(w)
        return [w[i] for i in idx]

        
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

class GBC(Model):
    def __init__(self, **kwargs):
        super().__init__()
        if len(kwargs) == 0:
            self.model = GradientBoostingClassifier()
        else:
            self.model = GradientBoostingClassifier(**kwargs)
    
    def fit(self):
        self.model.fit(self.Xtrain, self.Ytrain, self.train_weight)
        print(self.model.score(self.Xtrain, self.Ytrain))

    def predict(self):
        pred = self.model.predict(self.Xtest)
        ''' this is for checking the predict result
        for i in range(3):
            print("real %d, predict %d" % (sum(self.Ytrain==(i+1)), sum(pred==(i+1))))
            print(sum(((self.Ytrain==(i+1)) & (pred==(i+1)))))
        '''
        return pred

    def proba(self):
        return self.model.predict_proba(self.Xtest)