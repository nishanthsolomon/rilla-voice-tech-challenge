from modules.module import Embeddings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class OneHotEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_data(self, data):
        X = self.vectorizer.fit_transform(data)
        self.X = np.array(X.toarray())
        for a, b in zip(self.vectorizer.get_feature_names(), self.X.sum(axis=0)):
            print(a,'->',b)

    def get_embeddings(self, data):
        self.fit_data(data)
        return self.X
