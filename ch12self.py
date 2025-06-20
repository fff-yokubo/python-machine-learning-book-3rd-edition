
# %%
# # coding: utf-8
import os
import struct
import numpy as np


 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


X_train, y_train = load_mnist('mnist', kind='train')
print('TrainData, Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('mnist', kind='t10k')
print('TestData, Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# %%

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#NOTE: 画像表示

# fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
# ax = ax.flatten()
# for i in range(10):
#   img = X_train[y_train == i][0].reshape(28,28)
#   ax[i].imshow(img, cmap = 'Greys')

# ax[0].set_xticks([]);ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

# # %%


# #NOTE: 画像表示

# fig, ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
# ax = ax.flatten()
# for i in range(25):
#   img = X_train[y_train == 0][i].reshape(28,28)
#   ax[i].imshow(img, cmap = 'Greys')

# ax[0].set_xticks([]);ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()


# %%
#NOTE: numpyアーカイブとして保存
#np.savez_compressed('mnist/mnist_scled.npz', X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test)
# %%
mnist = np.load('mnist/mnist_scled.npz')

nnfile = "mnist_learned.pkl"

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
# %%
import pickle as pkl
from ch12.neuralnet import NeuralNetMLP



if os.path.exists(nnfile):

  print("LOAD: %s"%nnfile)
  with open(nnfile, 'rb') as f:
    nn = pkl.load(f)

else:
    
  print("Initialize: Neuralnet")


  nn = NeuralNetMLP(
    n_hidden = 100, 
    l2 = 0.01,
    epochs = 200,
    eta = 0.0005,
    minibatch_size = 100,
    shuffle = True,
    seed = 1
  )
  nn.fit(
    X_train = X_train[:55000], y_train = y_train[:55000], X_valid = X_train[55000:], y_valid =y_train[55000:])

  with open(nnfile, 'wb') as f:
    pkl.dump(nn,f)
  print("Write result %s"%nnfile)


import matplotlib.pyplot as plt

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('cost')
plt.xlabel('Epochs')

plt.show()


# %%
