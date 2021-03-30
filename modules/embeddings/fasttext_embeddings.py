from modules.module import Embeddings
import numpy as np
import fasttext


class FasttextEmbeddings(Embeddings):
    def __init__(self, clean=False):
        self.clean = clean

    def fit_data(self):
        if self.clean:
            self.model = fasttext.train_unsupervised(
                '../../data/fasttext_data.txt', minn=2, maxn=5, dim=100, epoch=200, lr=0.5, thread=6)
        else:
            self.model = fasttext.train_unsupervised(
                '../../data/fasttext_cleaned_data.txt', minn=2, maxn=5, dim=100, epoch=200, lr=0.5, thread=6)

    def get_embeddings(self, data):
        self.fit_data()
        embeddings = []
        for sent in data:
            embedding = np.array([self.model.get_word_vector(x)
                                  for x in sent.split()])
            embeddings.append(embedding.mean(axis=0))
        return np.array(embeddings)
