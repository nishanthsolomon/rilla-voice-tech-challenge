from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def visualize_model(X, assigned_clusters):
    NUM_CLUSTERS = len(set(assigned_clusters))
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=[
        'principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, pd.DataFrame(
        assigned_clusters, columns=['target'])], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [i for i in range(NUM_CLUSTERS)]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][:NUM_CLUSTERS]
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    fig.show()
    input('press return to continue')


def get_silhouette_score(X, cluster_labels):
    return silhouette_score(X, cluster_labels)