from sklearn import mixture
from sklearn import cluster
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import copy
import tkinter


with open('dataset.txt', "r") as data_set:
    data = pd.read_table(data_set, delim_whitespace=True,
                         header=None)


matrix = data.drop(7, axis=1).values

"""Following are all the parameters for the Gaussian clustering"""
# no default
numberOfClusters = 3
# default = 1
initializations = 1
# default = 100
iterations = 100
# default 1e-6
regularization = 1e-2
# default full
covarType = "spherical"
# default 1e-3
covarTreshold = 1e-3
# default "kmeans"
method = "kmeans"
# optional
initialWeights = [.3, .3, .4]
# optional
means = None
# optional
initialPrecisions = None
# optional
seed = None
# default false
warm = False


def TSNE_dimensionality_reduction(data_matrix):
    return TSNE(n_components=2, perplexity=12, early_exaggeration=14, learning_rate=100).fit_transform(data_matrix)


def PCA_dimensionality_reduction(data_matrix):
    return PCA(n_components=2).fit_transform(data_matrix)


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


def makeVis(data, classes, tittel, idx):
    plt.subplot(2, 1, idx)

    plt.subplots_adjust(
        hspace=.4,
        wspace=.2,
        top=.95
    )
    plt.title(tittel)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(data[:, 0], data[:, 1],
                s=50,
                c=classes,
                alpha=0.5,
                edgecolors='none'
                )
    # print(data)


def elbowMethod(data_matrix):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = cluster.KMeans(n_clusters=k).fit(data_matrix)
        kmeanModel.fit(data_matrix)
        distortions.append(sum(np.min(
            cdist(data_matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data_matrix.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortions')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def numberInEachCluster(title_, clust):
    flat_ = list(clust.flatten())
    output = f""" {title_}
                 Original Category 1: 
                 Cluster 1: {flat_[0:70].count(0)} units, Cluster 2: {flat_[0:70].count(1)} units, Cluster 3: {flat_[0:70].count(2)}
                 Original Category 2:
                 Cluster 1: {flat_[70:140].count(0)} units, Cluster 2: {flat_[70:140].count(1)} units, Cluster 3: {flat_[70:140].count(2)}
                 Original Category 3;
                 Cluster 1: {flat_[140:210].count(0)} units, Cluster 2: {flat_[140:210].count(1)} units, Cluster 3: {flat_[140:210].count(2)}
                 """

    return output


def main():

    copy_one = copy.deepcopy(matrix)
    copy_two = copy.deepcopy(matrix)
    copy_three = copy.deepcopy(matrix)
    copy_four = copy.deepcopy(matrix)
    elbowMethod(copy_four)

    kMeans_clustered = kMeans_clustering(copy_one)
    gauss_clustered = gaussianClustering(copy_two)
    reduced_matrix = TSNE_dimensionality_reduction(copy_three)
    #reduced_matrix = PCA_dimensionality_reduction(copy_three)

    makeVis(reduced_matrix, kMeans_clustered, "K_means", 1)
    makeVis(reduced_matrix, gauss_clustered, "Gauss", 2)
    plt.show()


if __name__ == "__main__":
    main()
