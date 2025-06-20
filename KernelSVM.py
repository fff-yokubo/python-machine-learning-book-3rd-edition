# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
import common
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

moons = make_moons(n_samples = 1000, noise = 0.05, random_state = 123)

X = moons[0]
y = moons[1]


kernel_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',SVC(kernel = 'rbf'))
  ])

kernel_svm.fit(X,y)


plt.figure(figsize=(12,8))
# common.plot_decision_function(kernel_svm)
common.plot_dataset(X,y)
plt.show()
# %%
