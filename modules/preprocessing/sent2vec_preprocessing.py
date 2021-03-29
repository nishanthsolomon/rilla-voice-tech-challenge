from modules.module import Preprocessing
import pandas as pd


class Sent2VecPreprocessing(Preprocessing):
    def __init__(self):
        pass

    def read_data(self, path='../../data/Rilla NLP Dataset - Sheet1.csv'):
        self.data = pd.read_csv(path)

    def get_data(self):
        self.read_data()

        return self.data
