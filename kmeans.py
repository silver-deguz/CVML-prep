import numpy as np 
from sklearn.datasets import make_blobs
from collections import defaultdict



class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = [0] * self.k
        self.clusters = [[] for _ in range(self.k)]

        self.data = None

    def init_centroids(self):
        assert self.data is not None 
        idxs = np.random.randint(0, len(self.data), size=self.k)
        for i in range(self.k):
            self.centroids[i] = self.data[idxs[i]]
    
    def update_clusters(self, centroids):
        # clear previous clusters
        self.clusters = [[] for _ in range(self.k)] 
        for i, x in enumerate(self.data):
            dist_k = [ self._euclidean_distance(x - mu) for mu in centroids ]
            idx = np.argmin(dist_k)
            self.clusters[idx].append(i)

    def update_centroids(self):
        self.centroids = [0] * self.k
        for idx, cluster in enumerate(self.clusters):
            new_mu = np.mean(self.data[cluster], axis=0)
            self.centroids[idx] = new_mu
    
    def _euclidean_distance(self, x):
        return np.linalg.norm(x)

    def run_clustering(self, data):
        self.data = data
        self.init_centroids()
        N = self.data.shape[0]

        converged = False
        num_iter = 0
        while not converged and num_iter < self.max_iter:
            mu_old = self.centroids.copy()
            # update clusters based on euclidean distance
            self.update_clusters(self.centroids)
        
            # update centroids
            self.update_centroids()
            mu_new = self.centroids

            # check for convergence
            deltas = [self._euclidean_distance(mu1 - mu2) < 0.0001 for mu1, mu2 in zip(mu_new, mu_old)]
            if all(deltas):
                print(self.centroids)
                num_points_in_clusters = [len(cluster) for cluster in self.clusters] 
                print(num_points_in_clusters)
                converged = True
            
            num_iter+=1

            
def elbow_curve():
    pass
            
def main():
    np.random.seed(15)
    points = make_blobs(n_samples=200, n_features=3, centers=4, cluster_std=.7)
    data, labels = points[0], points[1]

    k_means = KMeans(k=3)
    k_means.run_clustering(data)