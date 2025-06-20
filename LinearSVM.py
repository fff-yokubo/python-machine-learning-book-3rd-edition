
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
import common
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

moons = make_moons(n_samples = 100000, noise = 0.1, random_state = 0)

X = moons[0]
y = moons[1]


poly_svm = Pipeline([
    ('poly', PolynomialFeatures(degree = 30)),
    ('scaler', StandardScaler()),
    ('svm',LinearSVC())
  ])

poly_svm.fit(X,y)


plt.figure(figsize=(12,8))
common.plot_decision_function(poly_svm)
common.plot_dataset(X,y)
plt.show()