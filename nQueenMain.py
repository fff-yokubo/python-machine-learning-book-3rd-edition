

import gurobipy as gp
from nQueenSub import nQueenModel



nsize = 10


for i in range(1,nsize+1):

  #N-Queen問題インスタンス
  m = nQueenModel(tableSize = i)
  nct = 0

  while m.checkFeasibility():
    m.setPrevConstr()
    nct += 1
  print("Nsize = %s, SolCount = %s"%(i,nct))  


