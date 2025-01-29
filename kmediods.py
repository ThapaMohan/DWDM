import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

# Generate synthetic data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3, random_state=42)

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=42).fit(X)

# Labels for each data point
labels = kmedoids.labels_

# Plot the clusters and their medoids
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    # Get the points in the cluster
    class_member_mask = labels == k
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)

# Plot the medoids
plt.plot(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], "o", markerfacecolor="cyan", markeredgecolor="k", markersize=10)

# Add title and display the plot
plt.title("K-Medoids Clustering (Medoids in Cyan)")
plt.show()
