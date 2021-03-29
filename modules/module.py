class Preprocessing():
    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def save_data(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError


class Embeddings():
    def fit_data(self):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError


class Clustering():
    def get_clusters(self):
        raise NotImplementedError

    def set_model(self):
        raise NotImplementedError

    def fit_model(self):
        raise NotImplementedError

    def generate_metrics(self):
        raise NotImplementedError

    def predict_model(self):
        raise NotImplementedError
