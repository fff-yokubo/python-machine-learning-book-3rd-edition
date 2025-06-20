# %%


import pandas as pd

df = pd.read_csv("ch10/housing.data.txt", header=None, sep ="\s+")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from mlxtend.plotting import scatterplotmatrix

cols = ["LSTAT","INDUS","NOX","RM","MEDV"]

scatterplotmatrix(df[cols].values, figsize = (10,8), names = cols , alpha = 0.5)
plt.tight_layout()
plt.show()

# %%
