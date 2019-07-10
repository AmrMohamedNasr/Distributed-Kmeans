from sklearn.cluster import KMeans
import pandas as pd 
import numpy as np
import time
from sklearn import datasets

iris = datasets.load_iris()
start = time.time()
data = iris['data']
values = data
data_no_label = values[:,0:4]
kmeans = KMeans(n_clusters=3, random_state=0, n_jobs=1).fit(data_no_label)
print(kmeans.cluster_centers_)
end = time.time()
print("Time taken :", end - start)
