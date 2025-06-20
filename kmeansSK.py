# %%
import numpy as np
from sklearn.datasets import make_blobs

n_centers=3
n_features=2
n_samples=300
#X: 特徴量ベクトル
#y: クラスタ(正解)
X, y = make_blobs(
  n_samples = n_samples,#NOTE: データ点の総数
  n_features = n_features, #NOTE: 特徴量の個数,
  centers = n_centers, #クラスタの個数,
  cluster_std = 0.5, #クラスタ内の標準偏差,
  shuffle = False, #データ点のシャッフル
  random_state = 0#乱数発生機の状態
)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.cluster import KMeans

km = KMeans(
  n_clusters = 3,
  init = 'k-means++',
  n_init = 10,
  max_iter = 300,
  tol = 1e-04,
  random_state = 0
)

y_km = km.fit_predict(X)
# %%


plt.scatter(
  X[y_km==0, 0],X[y_km==0, 1],
  s = 50,
  c= 'lightgreen',
  edgecolor = 'black',
  marker = 's',
  label = 'Cluster 1'
)
plt.scatter(
  X[y_km==1, 0],X[y_km==1, 1],
  s = 50,
  c= 'orange',
  edgecolor = 'black',
  marker = 'o',
  label = 'Cluster 2'
)

plt.scatter(
  X[y_km==2, 0],X[y_km==2, 1],
  s = 50,
  c= 'lightblue',
  edgecolor = 'black',
  marker = 'v',
  label = 'Cluster 3'
)

plt.scatter(
  km.cluster_centers_[:,0],
  km.cluster_centers_[:,1],
  s = 250,
  marker= '*',
  c = "red",
  edgecolor = 'black',
  label = 'Centroids'
)


plt.legend(scatterpoints = 1)
plt.grid()
plt.tight_layout()
plt.show()

# %%


import pandas as pd

np.random.seed(123)

n_points = 5

variables = ['X','Y','Z']

labels = ["ID_%d"%d for d in range(n_points)]

X = np.random.random_sample([n_points,3])*10

df = pd.DataFrame(X, columns = variables, index = labels)

df
# %%

from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(
  squareform(pdist(df, metric ='euclidean')),
    columns = labels, index =labels
    )

row_dist

# %%
