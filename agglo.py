#agglomerative clustering algorithm
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plot
x = [[i] for i in [12,10,6,8,4,2,1,16]]
Z = linkage (x, 'ward')
figure =plot.figure(figsize=(25,10))
den = dendrogram(Z)
Z=linkage(x,'single')
figure = plot.figure(figsize=(25,10))
den = dendrogram(Z)
plot.show()