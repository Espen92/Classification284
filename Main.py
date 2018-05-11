from sklearn import mixture
from sklearn import cluster
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open('dataset.txt', "r") as data_set:
    data = pd.read_table(data_set, delim_whitespace=True,
                         header=None)


matrix = data.drop(7, axis=1).values

# gauss params
# no default
numberOfClusters = 3
#default = 1
initializations = 1
#default = 100
iterations = 100
# default 1e-6
regularization = 1e-2
# default full
covarType = "full"
# default 1e-3
covarTreshold = 1e-3
# default "kmeans"
method = "kmeans"
# optional
initialWeights = None
# optional
means = None
# optional
initialPrecisions = None
# optional
seed = None
# default false
warm = False


def TSNE_dimensionality_reduction(data_matrix):
    return TSNE(n_components=2).fit_transform(data_matrix)


def gaussianClustering(data_matrix):
    gm = mixture.GaussianMixture(
        n_components=numberOfClusters,
        covariance_type=covarType,
        tol=covarTreshold,
        n_init=initializations,
        max_iter=iterations,
        reg_covar=regularization,
        init_params=method,
        weights_init=initialWeights,
        means_init=means,
        precisions_init=initialPrecisions,
        random_state=seed,
        warm_start=warm
    )
    gm.fit(data_matrix)
    gauss_classes = gm.predict(data_matrix)
    return gauss_classes


def kMeans_clustering(data_matrix):
    k = cluster.KMeans(n_clusters=numberOfClusters)
    return k.fit_predict(data_matrix)


def makeVis(data, classes, tittel):
    plt.figure()
    plt.title(tittel)
    plt.xlabel('x LABEL')
    plt.ylabel('y LABEL')
    plt.scatter(data[:, 0], data[:, 1], s=(100), c=classes, alpha=0.5)


reduced_matrix = TSNE_dimensionality_reduction(matrix)
kMeans_clustered = kMeans_clustering(matrix)
gauss_clustered = gaussianClustering(matrix)
makeVis(reduced_matrix, kMeans_clustered, "K_means")
makeVis(reduced_matrix, gauss_clustered, "Gauss")
plt.show()
