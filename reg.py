# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()



x = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

from scipy.spatial.distance import pdist, squareform

y=pdist(x)

squareform(y)
# %%
squareform([0,1,2])
# %%
