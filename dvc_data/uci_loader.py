import uci_reader
import pandas as pd
class Loader():
    def __init__(self,encode_y=False):
        self.names = []
        self.drop = ['ID']
        self.encode_tag = []
        self.label_tag = 'Y'
        self.normalize_tag = []
        self.encode_y = encode_y
    def load(self):
        uci = uci_reader.Reader("", self.names, self.label_tag, drop_tags=self.drop, 
                encode_tags=self.encode_tag, normal_tags=self.normalize_tag, encode_y=self.encode_y)
        return uci()
    def __call__(self):
        return self.load()
