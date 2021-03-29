from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from modules.preprocessing.tf_idf_preprocessing import TfIdfPreprocessing
from modules.embeddings.one_hot_embeddings import OneHotEmbeddings
from nltk.cluster import KMeansClusterer
import nltk
from utils.utils import visualize_model


class OneHotClustering():
    def __init__(self):
        clusters_range = (2, 8)
        model = KMeans()
        self.visualizer = KElbowVisualizer(model, k=clusters_range)

        tf_idf_preprocessing = TfIdfPreprocessing()
        self.data = tf_idf_preprocessing.get_data()

        one_hot_embeddings = OneHotEmbeddings()
        self.X = one_hot_embeddings.get_embeddings(
            self.data['cleaned_content'])

        self.get_clusters()
        self.set_model()

    def get_clusters(self):
        self.visualizer.fit(np.array(self.X))
        # self.visualizer.show()

        self.NUM_CLUSTERS = self.visualizer.elbow_value_

        if self.NUM_CLUSTERS is None:
            self.NUM_CLUSTERS = 4
        print('Number of clusters: ', self.NUM_CLUSTERS)

    def set_model(self):
        self.kclusterer = KMeansClusterer(
            self.NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)

    def fit_model(self):
        self.assigned_clusters = self.kclusterer.cluster(
            self.X, assign_clusters=True)

    def predict_model(self):
        for j in range(self.NUM_CLUSTERS):
            for a, b in zip(self.data.Content.to_list(), self.assigned_clusters):

                if int(b) == j:
                    print(b, '->', a)

            print('\n\n')
        
        visualize_model(self.X, self.assigned_clusters)


if __name__ == '__main__':
    one_hot_clustering = OneHotClustering()
    one_hot_clustering.fit_model()
    one_hot_clustering.predict_model()
