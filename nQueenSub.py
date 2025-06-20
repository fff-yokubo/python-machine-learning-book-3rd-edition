import gurobipy as gp
import pprint as pp
import sys

import numpy as np
class nQueenModel(gp.Model):
  
  def __init__(self, tableSize):

    '''N-Queen Problem Class: arg: tableSize'''
    
    self._nsize = tableSize 
    super().__init__('%s -Queen Model'%(self._nsize))
    
    #NOTE: 決定変数 座標 0<= i,j <= N-1にQueenを配置する/しないと1/0
    self._x = {(i,j):self.addVar(vtype="B", name = "x[%s,%s]"%(i,j))
          for i in range(self._nsize) for j in range(self._nsize)}

    #NOTE: 盤面-縦(横)線上にQueenは1つまで
    for i in range(self._nsize): 
      xsumJ = gp.quicksum(self._x[i,j] for j in range(self._nsize))
      self.addConstr(xsumJ == 1, name = "Constr1V[%s]"%(i))
      xsumI = gp.quicksum(self._x[j,i] for j in range(self._nsize))
      self.addConstr(xsumI == 1, name = "Constr1H[%s]"%(i))

    #対角線方向(右上)にQueenは1まで
    origins = [(0,0)]
    for i in range(self._nsize-2):
      origins.extend([(i+1,0),(0,i+1)])
    for org in origins:  #原点について繰り返し
      xsumUp = gp.quicksum(self._x[org[0]+i, org[1]+i] #原点から右上方向、盤面が続く限り
                for i in range(self._nsize) if (org[0]+i, org[1]+i) in self._x)
      self.addConstr(xsumUp <= 1, name = "diagUp[%s,%s]"%(org[0],org[1]))

    #対角線方向(右下)にQueenは1つまで
    origins = [(0,self._nsize-1)]
    for i in range(self._nsize-2):
      origins.extend([(i+1,self._nsize-1),(0,self._nsize-i-2)])
    for org in origins:  
      xsumDn = gp.quicksum(self._x[org[0]+i, org[1]-i] #原点から右下方向、盤面が続く限り
                for i in range(self._nsize) if (org[0]+i,org[1]-i) in self._x)
      self.addConstr(xsumDn <= 1, name = "diagDn[%s,%s]"%(org[0],org[1]))

    #QueenがN(=盤面サイズ)こ配置できること
    self.addConstr(gp.quicksum(self._x.values()) == self._nsize, name = "nQueen")
    self.params.OutputFlag = 0 #gurobiログ非表示

  def displayResult(self):
    '''盤面表示テスト用 '''
    resultBan = [["◯" if self._x[self._nsize-i-1,j].X > 0.5 else "-" for i in range(self._nsize)]
      for j in range(self._nsize)]
    pp.pprint(resultBan)

  def checkFeasibility(self):
    '''Feasiblity Checker'''
    self.optimize() 
    if self.Status == gp.GRB.OPTIMAL:
      return True
    else:
      return False

  def setPrevConstr(self):
    '''前回以前と同じ盤面を返さない制約を追加'''
    Prev = [(i,j) for (i,j), xv in self._x.items() if xv.X > 0.5 ]
    xsum_Prev = gp.quicksum(self._x[i,j] for (i,j) in Prev)
    self.addConstr(xsum_Prev <= self._nsize-1)

if __name__ == '__main__':

  nsize = 10
  for i in range(1,nsize+1):
    m = nQueenModel(tableSize = i) # i-Queen問題インスタンス
    nct = 0 #パタン数(カウンタ)
    while m.checkFeasibility():#実行可能解が見つかる限り〃
      m.setPrevConstr()#前回と同じ解を出さない制約
      nct += 1#カウンタ
    print("Nsize = %s, SolCount = %s"%(i,nct))  
