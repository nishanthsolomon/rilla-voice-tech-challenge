from modules.module import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfIdfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_data(self, data):
        X = self.vectorizer.fit_transform(data)
        self.X = np.array(X.toarray())

    def get_embeddings(self, data):
        self.fit_data(data)
        return self.X
