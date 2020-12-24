import uci_loader
import pandas as pd
class Info():
    def __init__(self):
        self.url = "report.csv"
        self.names = ["%d" % x for x in range(21)]
        self.drop = ['0','1','4','5','6','7','11','12','14','15','16','18','19','20']
        self.encode_tag = ['2','3']
        self.label_tag = '10'
        self.normalize_tag = []
    def load(self):
        uci = uci_loader.Loader(self.url, self.names, self.label_tag, drop_tags=self.drop, 
                encode_tags=self.encode_tag, normal_tags=self.normalize_tag)
        return uci()
    def __call__(self):
        return self.load()
