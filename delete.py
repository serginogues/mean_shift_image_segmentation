from mean_shift import meanshift
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import MeanShift, estimate_bandwidth

X, y = make_blobs(n_samples=1000, n_features = 3, centers = [(5,5), (3,3), (1,1)], cluster_std = 0.30)

#plt.scatter(X[:, 0], X[:, 1])

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
"""meanshift_2 = MeanShift(bandwidth=bandwidth)
meanshift_2.fit(X)

labels__ = meanshift_2.labels_
n_clusters_ = len(np.unique(labels__))
print('Estimated number of clusters: ' + str(n_clusters_))

y_pred  = meanshift_2.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis")"""

labels, peaks = meanshift(X, r=bandwidth, c=4)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")

plt.scatter(peaks[:, 0], peaks[:, 1], c='r', cmap="viridis")
plt.show()

