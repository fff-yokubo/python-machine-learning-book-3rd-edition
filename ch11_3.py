# %%

#NOTE: DBSCAN: Density Based Spatial Clustering of Applications with Noise
#指定された変形e以内に存在する点の個数


#・指定された半径e以内に少なくおｔも指定された個数(MinPts)の隣接点がるような点は、コア点(core point)とみなす
#半径e以内の隣接店の個数がMinPrsに満たないものの、コア店の半径e以内に位置する位置するような点は、ボーダー点とみなす。
#それ以外(コア点、ボーダー点)はノイズ点とみなす。

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_moons



X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)


# plt.scatter(X[:,0],X[:,1])
# plt.tight_layout()
# plt.show()

# %%
# k-means法と完全連結法の比較

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

f, (ax1, ax2,ax3) = plt.subplots(nrows=1,ncols=3, figsize = (30,8))

km = KMeans(n_clusters = 2, random_state = 0)

cs = ['lightblue', 'red']
ms = ['o', 's']


#NOTE: KMeans法
y_km = km.fit_predict(X)
for idx, (c, m) in enumerate(zip(cs, ms)):

  ax1.scatter(X[y_km==idx,0],X[y_km==idx, 1],c = c, edgecolor = 'black', marker = m, s=40, label = 'cluster %s'%(idx+1))
ax1.set_title('K-means clustering')

#NOTE* 凝集型クラスタリング(完全連結法)

ac = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'complete')

y_ac = ac.fit_predict(X)
for idx, (c, m) in enumerate(zip(cs, ms)):

  ax2.scatter(X[y_ac==idx,0],X[y_ac==idx, 1],c = c, edgecolor = 'black', marker = m, s=40, label = 'cluster %s'%(idx+1))
ax2.set_title('Agglomerative(凝集型) clustering')

db = DBSCAN(eps = 0.2, min_samples=5, metric = 'euclidean')

y_db = db.fit_predict(X)
for idx, (c, m) in enumerate(zip(cs, ms)):

  ax3.scatter(X[y_db==idx,0],X[y_db==idx, 1],c = c, edgecolor = 'black', marker = m, s=40, label = 'cluster %s'%(idx+1))
ax3.set_title('DBSCAN clustering')

plt.show()
# %%
