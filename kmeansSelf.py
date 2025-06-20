
import numpy as np



class kMeans:
  
  def __init__(self,X, n_centers,max_iteration = 100,ise = 1e-5, rseed = 10):
    
    #1. クラスタ中心の初期値として、データ点からNclusterこのセントロイドをランダムに選択する。
    np.random.seed(seed=rseed)
    #NOTE: n_samplesこのデータ点より、n_centers個をランダムに選択する。シード値を設定して毎回同じ値にする。
    self.n_centers = n_centers
    self._X = X
    n_samples = X.shape[0]
    self._centroids = X[np.random.choice(n_samples, n_centers, replace = False),:]


    self._max_iteration = max_iteration
    self._ise = ise

  def fit1step(self):
    
    #2: 各データ点を最も近いセントロイドに割り当てる
    #データiごとに、下記を判定
    # 距離=np.linalg.norm(a-b)

    #データ点i(Xi)ごとに、セントロイドjとのユークリッド距離を算出(np.linalg.norm)。
    # 距離が最も近いセントロイドを選択する(np.argmin)
    results = [
      np.argmin([np.linalg.norm(Xi-c_j) for c_j in self._centroids])
      for Xi in self._X
    ]


    #NOTE: セントロイドnに割当られたデータの中心にセントロイドを移動する
    self._centroids =[
      #jに属するデータ点をチョイスして、その中心(np.mean)を新たな中心点とする。
      np.mean(self._X[[idx for idx, c in enumerate(results) if c == j],:], axis = 0).tolist()
      for j in range(self.n_centers)
    ]

    #gosa
    
    self._mse = sum([
      np.linalg.norm(results[idx]-Xi) 
      for idx, Xi in enumerate(self._X)
    ])
    
    self.results = np.array(results)
    

  def fit_transform(self):
    
    '''
    指定回数もしくは、収束率が規定を超過すると中断
    '''

    cresult = []
    mse_old = -1
    for i in range(self._max_iteration):

      self.fit1step()
      cresult.append(self._centroids)

      mse_new = self._mse

      if i==0:
        mse_old = mse_new
        continue

      mse_chg = np.abs(mse_new - mse_old) / np.abs(mse_new)
      if mse_chg < self._ise:
        break
      
      mse_old = mse_new
    else:
      print("Maximum iteration exceeded")
    
    
    self.centroids = cresult
    self.mse = mse_new


if __name__ == '__main__':


  from sklearn.datasets import make_blobs

  n_centers=4
  n_features=2
  n_samples=10000
  #X: 特徴量ベクトル
  #y: クラスタ(正解)
  X, y = make_blobs(
    n_samples = n_samples,#NOTE: データ点の総数
    n_features = n_features, #NOTE: 特徴量の個数,
    centers = n_centers, #クラスタの個数,
    cluster_std = 1.5, #クラスタ内の標準偏差,
    shuffle = False, #データ点のシャッフル
    random_state = 0#乱数発生機の状態
  )

  import matplotlib.pyplot as plt
  import seaborn as sns; sns.set()



  model = kMeans(X, n_centers = 4, max_iteration = 200, ise = 1e-10, rseed = 10)

  model.fit_transform()
  
  

  # centroids = []
  # mse = []
  # for _ in range(10):
  #   model.fit1()
  #   centroids.append(model._centroids)
  #   mse.append(model._mse)

  colors = ['green','red','blue','black']

  centroids = model.centroids

  for idx, color in enumerate(colors):
    plt.scatter(X[model.results==idx,0],X[model.results==idx,1], c = color, marker = 'o', s = 10, alpha = 0.2)
    ci = np.array(centroids)[:,idx,:].T
    plt.scatter(ci[0],ci[1], c = color, marker = '+', s = 300, alpha = 1)
    plt.plot(ci[0],ci[1], c = color, linestyle = 'solid')


  plt.grid()


  plt.tight_layout()
  plt.show()

  print("mse=",model.mse)