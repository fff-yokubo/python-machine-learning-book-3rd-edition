from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()


import numpy as np

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.7, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
        
    plt.xlabel('主成分1');plt.ylabel('主成分2');plt.legend(loc='best')
    
    
def plot_decision_function(model):
  _x0 = np.linspace(-1.5, 2.5, 100)
  _x1 = np.linspace(-1.0, 1.5, 100)
  x0, x1 = np.meshgrid(_x0,_x1)
  
  X = np.c_[x0.ravel(), x1.ravel()]
  y_pred = model.predict(X).reshape(x0.shape)
  
  y_decision = model.decision_function(X).reshape(x0.shape)
  
  plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha = 0.2)
  plt.contourf(x0, x1, y_decision, levels = [y_decision.min(), 0, y_decision.max()], alpha =0.3)
  
def plot_dataset(X, y):
  plt.plot(X[:,0][y==0], X[:, 1][y==0], "o", alpha = 0.6)
  plt.plot(X[:,0][y==1], X[:, 1][y==1], "^", alpha = 0.6)
  
  plt.xlabel("$x_1$", fontsize=20)
  plt.ylabel("$x_2$", fontsize=20, rotation = 0)
  

def lin_regplot(X,y, model):
  plt.scatter(X,y, c='steelblue', edgecolor = 'white', s = 70)
  plt.plot(X, model.predict(X), color = 'black', lw = 2)
  
  return None