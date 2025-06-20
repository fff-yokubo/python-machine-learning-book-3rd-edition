
# %%
import numpy as np

import pandas as pd

np.random.seed(123)

n_points = 5

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
import seaborn as sns; sns.set()

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
