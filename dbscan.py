#DBSCAN
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from IPython.display import display_html
def toy_dataset():
    value = np.array([[1,5],[3,2], [4,2],[7,7],[9,2], [13,14], [12,11],
                        [55,15], [30,30],[54,18], [49,19],[52,20],
                        [54,12],[43,29], [49,19],[47,52],[51,27],
                        [52,49],[42,43], [43,46], [49,52],[51,48],
                        [53,60],[51,50],[52,53], [51,51],[55,53],
                        [25,70], [90,32], [120,70]])
    titles = ['x', 'y']
    data = pd.DataFrame(value, columns=titles)
    data.plot.scatter(x='x',y='y')
    return data
def Dbscan_clustering(data):
    db = DBSCAN(eps=10.5,min_samples=4).fit(data)
    core_samples_mask = np. zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_]=True
    labels = pd.DataFrame(db.labels_, columns=['Cluster ID'])
    result = pd.concat((data, labels), axis=1)
    result.plot.scatter (x='x',y='y',c='Cluster ID', colormap='jet')
def main():
    data = toy_dataset()
    print("DBSCAN algo implementation")
    Dbscan_clustering(data)
main()