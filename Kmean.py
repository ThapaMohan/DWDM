#K-Means Clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = np.random.rand(600, 2) * 100
km = KMeans (n_clusters=4, init="random")
km.fit(data)
centers=km.cluster_centers_
labels =km.labels_
print("Cluster Centers:", centers)
colors = ["g", "r", "b", "y", "m"]
markers = ["+", "x", "*", ".","d"]
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[labels[i]], marker=markers[labels[i]])
plt.scatter(centers[:, 0], centers[:, 1], marker="o", s=50, linewidths=5)
plt.show()