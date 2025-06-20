
# %%
import numpy as np

import pandas as pd

np.random.seed(123)

n_points = 10

variables = ['X','Y','Z']

labels = ["ID_%d"%d for d in range(n_points)]

X = np.random.random_sample([n_points,3])*10

df = pd.DataFrame(X, columns = variables, index = labels)

df
df.style.background_gradient()

# %%

from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(
  squareform(pdist(df, metric ='euclidean')),
    columns = labels, index =labels
    )

row_dist
row_dist.style.background_gradient()

pdist(df, metric='euclidean')
# %%

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
# row_clusters = linkage(
#   pdist(df, metric = 'euclidean'), method = 'complete', metric = 'euclidean'
# )

row_clusters = linkage(
  df.values, method = 'complete'
)


# %%

result = pd.DataFrame(row_clusters,
             columns = ['row_label_1','row_label_2', 'distance', "no. of items in cliuster"],
             index = ['cluster %d'%(i+1) for i in range(row_clusters.shape[0])]
            )
result

# %%
set_link_color_palette(['black'])

row_dendr = dendrogram(
  row_clusters, labels = labels
)

plt.ylabel('Euclidean distance')
plt.tight_layout()
plt.show()

# %%
#NOTE: 樹形図とヒートマップの組み合わせ
#NOTE: 新しいfigureオブジェクトを作成し、add_axes属性を使って樹形図のx,y軸の位置、幅、高さを定義する。さらに、樹形図を反時計回りに90度回転させる

fig = plt.figure(figsize=(8,8), facecolor = 'white')

axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])#x軸位置、y軸位置、幅、高さ

row_dendr = dendrogram(row_clusters, orientation = 'left')
#NOTE: クラスタリングのラベルに従って、最初のDataFrameObjectのデータを並び替え。

df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax= axm.matshow(df_rowclust, interpolation = 'nearest', cmap = 'hot_r')
# %%

#NOTE; 外観を整える: 軸の目盛を取り除き、縦横の軸を非表示にする
#カラーバーを追加してｘ軸の目盛ラベルとして、特徴量、y軸の目盛ラベルとしてデータ点の名前を割り当てる。
axd.set_xticks([])
axd.set_yticks([])

for i in axd.spines.values():
  i.set_visible(False)

fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))

plt.show()

# %%
