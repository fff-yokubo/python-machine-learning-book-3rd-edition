import gurobipy as gp
import pprint as pp
import sys

import numpy as np
class nQueenModel(gp.Model):
  
  def __init__(self, tableSize):

    '''
    N-Queen Problem Class
    arg: tableSize
    '''
    
    self._nsize = tableSize#セクションの数 サイズの平方根(4x4⇢2, 9x9 ⇢3, 16x16⇢n) 
    name = '%s -Queen Model'%(self._nsize)
    super().__init__(name)
    
    #NOTE: 決定変数 座標 0<= i,j <= N-1にQueenを配置する/しないと1/0
    self._x = {(i,j):self.addVar(vtype="B", name = "x[%s,%s]"%(i,j))
          for i in range(self._nsize) for j in range(self._nsize)}

    #NOTE: 縦横方向(飛車)
    for i in range(self._nsize): 
      xsumJ = gp.quicksum(self._x[i,j] for j in range(self._nsize))
      self.addConstr(xsumJ == 1, name = "Constr1V[%s]"%(i))
      xsumI = gp.quicksum(self._x[j,i] for j in range(self._nsize))
      self.addConstr(xsumI == 1, name = "Constr1H[%s]"%(i))

    #対角線方向(右上)
    origins = [(0,0)]
    for i in range(self._nsize-2):
      origins.append((i+1,0))
      origins.append((0,i+1))

    for org in origins:  
      xsum_points = gp.quicksum(self._x[org[0]+i, org[1]+i]
                for i in range(self._nsize)
                if (org[0]+i, org[1]+i) in self._x)
      self.addConstr(xsum_points <= 1, name = "diagUp[%s,%s]"%(org[0],org[1]))

    #対角線方向(右下)

    origins = [(0,self._nsize-1)]
    for i in range(self._nsize-2):
      origins.append((i+1,self._nsize-1))
      origins.append((0,self._nsize-i-2))

    for org in origins:  
      xsum_points = gp.quicksum(self._x[org[0]+i, org[1]-i]
                for i in range(self._nsize)
                if 0 <= org[0]+i < self._nsize and 0<= org[1]-i < self._nsize
                )
      self.addConstr(xsum_points <= 1, name = "diagDn[%s,%s]"%(org[0],org[1]))

    self.addConstr(gp.quicksum(self._x.values()) == self._nsize, name = "nQueen")
    self.params.OutputFlag = 0

  def displayResult(self):
    '''
    盤面表示
    '''
    resultBan = [
      ["◯" if self._x[self._nsize-i-1,j].X > 0.5 else "-" for i in range(self._nsize)]
      for j in range(self._nsize)
    ]
    pp.pprint(resultBan)

  def checkFeasibility(self):
    '''
    Feasiblity Check
    '''
    self.optimize() 
    if self.Status == gp.GRB.OPTIMAL:
      return True
    else:
      return False

  def setPrevConstr(self):
    '''
    前回以前と同じ盤面を返さない制約を追加
    '''
    Prev = [(i,j) for (i,j), xv in self._x.items() if xv.X>0.5 ]
    xsum_result_Prev = gp.quicksum(self._x[i,j] for (i,j) in Prev)
    self.addConstr(xsum_result_Prev <= self._nsize-1)


