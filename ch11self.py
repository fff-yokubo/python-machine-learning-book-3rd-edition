from sklearn.datasets import make_blobs
import numpy as np
n_centers = 3
n_features = 2
n_samples  =150
#X: 特徴量ベクトル
#y: クラスタ(正解)
X, y = make_blobs(
  n_samples = n_samples,#NOTE: データ点の総数
  n_features = n_features, #NOTE: 特徴量の個数,
  centers = n_centers, #クラスタの個数,
  cluster_std = 0.5, #クラスタ内の標準偏差,
  shuffle = True, #データ点のシャッフル
  random_state = 0#乱数発生機の状態
)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.cluster import KMeans

km = KMeans(
  n_clusters=8,
  init = 'random',
  n_init = 10,
  max_iter= 300,
  tol = 1e-04,
  random_state = 0
)

y_km = km.fit_predict(X)

from matplotlib import cm

from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)

n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X, y_km, metric = 'euclidean')

y_ax_lower, y_ax_upper = 0,0

y_ticks = []

for i, c in enumerate(cluster_labels):
  c_silhouette_vals =silhouette_vals[y_km==c]
  c_silhouette_vals.sort()
  
  y_ax_upper += len(c_silhouette_vals)
  color = cm.jet(float(i)/ n_clusters)
  plt.barh(
    range(y_ax_lower, y_ax_upper),
    c_silhouette_vals,
    height = 1.0,
    edgecolor = 'none',
    color = color
    )

  y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
  y_ax_lower += len(c_silhouette_vals)
  
silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg, color = 'red', linestyle = '--')

plt.yticks(y_ticks, cluster_labels + 1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

print(cluster_labels)