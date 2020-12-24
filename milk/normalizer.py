import pandas as pd
import numpy as np
class Normalizer(object):
    def __init__(self,data = pd.DataFrame(), tags=[]):
        self.data = data
        self.tags = tags
        self.means = {}
        self.vars = {}
        self.mins = {}
        self.maxs = {}

    def __call__(self):
        return self.z_score()

    def set_data(self, data, tags):
        self.data = data
        self.tags = tags
        for t in self.tags:
            self.means[t] = self.data[t].mean()
            self.vars[t] = self.data[t].var()
            self.mins[t] = self.data[t].min()
            self.maxs[t] = self.data[t].max()

    def z_score(self):
        if(len(self.data) == 0):
            return []
        for t in self.tags:
            self.data[t] = (self.data[t] - self.means[t]) / np.sqrt(self.vars[t])
        return self.data

    def max_min(self):
        if(len(self.data) == 0):
            return []
        for t in self.tags:
            self.data[t] = (self.data[t] - self.mins[t]) / (self.maxs[t] - self.mins[t])
        return self.data
