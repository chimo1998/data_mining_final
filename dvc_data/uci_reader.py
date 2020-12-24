import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder as ohe, LabelBinarizer as LB
import numpy as np
from sklearn.model_selection import train_test_split as tts
from normalizer import Normalizer

class Reader(object):
    def __init__(self, url, names, label_tag, drop_tags=None, encode_tags=None, normalizer=Normalizer(), normal_tags=None, test_size=0.2, encode_y=False):
        self.url = url
        self.names = names
        self.drop_tags = drop_tags
        self.encode_tags = encode_tags
        self.data = None
        self.label_tag = label_tag
        self.test_size = test_size
        self.enc = ohe(categories='auto')
        self.normal_tags = normal_tags
        self.normalizer = normalizer
        self.split = 0
        self.train = None
        self.test = None
        self.encode_y = encode_y

    def __call__(self, normal=False):
        return self.load()

    def read_from_url(self):
        os.chdir(os.path.dirname(__file__))
        self.train = pd.read_csv(os.path.join(os.getcwd(),"train.csv"))
        self.test = pd.read_csv(os.path.join(os.getcwd(),"test.csv"))

    def drop(self):
        self.train = self.train.drop(self.drop_tags, axis=1)
        self.test = self.test.drop(self.drop_tags, axis=1)

    def encode(self):
        for tag in self.encode_tags:
            t = pd.DataFrame(data=(self.enc.fit_transform(self.data[tag].to_numpy().reshape(-1,1)).toarray()))
            names = t.columns.tolist()
            new_names = dict(zip(names, ["%s_%s" % (tag, x) for x in names]))
            t = t.rename(columns=new_names)
            self.data = pd.concat([self.data.drop(tag, axis=1), t], axis=1)

    def get_data(self):
        trainX = self.train.drop(self.label_tag, axis=1).sort_index()
        trainY = self.train[self.label_tag].sort_index()
        if self.encode_y:
            encoder = LB().fit(self.train[self.label_tag])
            trainY = pd.DataFrame(encoder.transform(self.train[self.label_tag]))
        #trainY = pd.DataFrame(data=(self.enc.fit_transform(self.train[self.label_tag].sort_index().to_numpy().reshape(-1,1)).toarray()))
        testX = self.test
        testY = []
        return trainX, trainY, testX, testY

    def normal(self):
        self.normalizer.set_data([self.data],self.normal_tags)
        self.normalizer()

    def load(self):
        self.read_from_url()
        if self.drop_tags:
            self.drop()
        if self.encode_tags:
            self.encode()
        if self.normal_tags:
            self.normal()
        return self.get_data()

    def is_num(self, v):
        result = True
        try:
            tmp = int(v)
            result = True
        except:
            result = False
        return result
