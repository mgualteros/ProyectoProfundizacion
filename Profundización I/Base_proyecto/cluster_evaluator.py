from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

class ClusterEvaluator:
    def __init__(self, data):
        self.data = data

    def evaluate_kmeans(self, k_range, seed=42):
        inertias, silhouettes, calinski_scores = [], [], []
        labels_dict = {}

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=seed)
            labels = model.fit_predict(self.data)
            inertias.append(model.inertia_)
            silhouettes.append(silhouette_score(self.data, labels))
            calinski_scores.append(calinski_harabasz_score(self.data, labels))
            labels_dict[k] = labels

        best_k = k_range[np.argmax(silhouettes)]

        return {
            'method': 'kmeans',
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouettes': silhouettes,
            'calinski': calinski_scores,
            'best_k': best_k,
            'best_labels': labels_dict[best_k]
        }

    def evaluate_dbscan(self, eps_values, min_samples=5):
        results = []
        for eps in eps_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(self.data)
            if len(set(labels)) > 1 and -1 not in set(labels):
                sil = silhouette_score(self.data, labels)
                calinski = calinski_harabasz_score(self.data, labels)
                results.append((eps, sil, calinski, labels))
        return results

    def plot_kmeans_metrics(self, k_values, inertias, silhouettes):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(k_values, inertias, marker='o', color='blue')
        axs[0].set_title('Inertia vs k')
        axs[1].plot(k_values, silhouettes, marker='o', color='green')
        axs[1].set_title('Silhouette vs k')
        plt.tight_layout()
        plt.show()