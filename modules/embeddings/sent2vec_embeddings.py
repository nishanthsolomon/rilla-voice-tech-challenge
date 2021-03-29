from modules.module import Embeddings
import numpy as np
from sent2vec.vectorizer import Vectorizer


class Sent2VecEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = Vectorizer()

    def fit_data(self, data):
        self.vectorizer.bert(data)
        self.X = self.vectorizer.vectors

    def get_embeddings(self, data):
        self.fit_data(data)
        return self.X
