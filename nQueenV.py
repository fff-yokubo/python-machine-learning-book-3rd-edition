import gurobipy as gp
import pprint as pp

class nQueenModel(gp.Model):
  def __init__(self, tableSize):
    '''N-Queen Problem Class: arg: tableSize'''
    self._nsize = tableSize
    super().__init__('%s -Queen Model'%(self._nsize))
    
    #NOTE: 決定変数 座標 0<= i,j <= N-1にQueenを配置する/しないと1/0 key=(i,j): 盤面座標
    self._x = {(i,j):self.addVar(vtype="B", name = "x[%s,%s]"%(i,j)) for i in range(self._nsize) for j in range(self._nsize)}

    #NOTE: 盤面-縦(横)線上にQueenは1つまで
    for i in range(self._nsize): 
      xsumJ = gp.quicksum(self._x[i,j] for j in range(self._nsize))#縦方向
      self.addConstr(xsumJ <= 1, name = "Constr1V[%s]"%(i))
      xsumI = gp.quicksum(self._x[j,i] for j in range(self._nsize))#横方向
      self.addConstr(xsumI <= 1, name = "Constr1H[%s]"%(i))

    #対角線方向(右上)にQueenは1まで
    origins = [(0,0)]
    for i in range(self._nsize-2):
      origins += [(i+1,0),(0,i+1)]
    for org in origins:  #原点について繰り返し
      xsumUp = gp.quicksum(self._x[org[0]+i, org[1]+i] #各原点から右上方向、盤面が続く限り
                for i in range(self._nsize) if (org[0]+i, org[1]+i) in self._x)
      self.addConstr(xsumUp <= 1, name = "ConstrUp[%s,%s]"%(org[0],org[1]))

    #対角線方向(右下)にQueenは1つまで
    origins = [(0,self._nsize-1)]
    for i in range(self._nsize-2):
      origins += [(i+1,self._nsize-1),(0,self._nsize-i-2)]
    for org in origins:  
      xsumDn = gp.quicksum(self._x[org[0]+i, org[1]-i] #各原点から右下方向、盤面が続く限り
                for i in range(self._nsize) if (org[0]+i,org[1]-i) in self._x)
      self.addConstr(xsumDn <= 1, name = "ConstrDn[%s,%s]"%(org[0],org[1]))

    #QueenがN(=盤面サイズ)こ配置できること
    self.addConstr(gp.quicksum(self._x.values()) == self._nsize, name = "nQueen")
    self.params.OutputFlag = 0 #gurobiログ非表示

  def dispResult(self):
    '''盤面表示テスト用 '''
    resultBan = [["◯" if self._x[self._nsize-i-1,j].X > 0.5 else "-" for i in range(self._nsize)]
      for j in range(self._nsize)]
    pp.pprint(resultBan)

  def checkFeas(self):
    '''Feasiblity Checker'''
    self.optimize() #求解。
    if self.Status == gp.GRB.OPTIMAL:#実行可能解あり⇢True
      return True
    return False#実行可能解なし⇢False

  def inhibitPrevSol(self):
    '''前回以前と同じ解を禁止する制約を追加'''
    Prev = [(i,j) for (i,j), xv in self._x.items() if xv.X > 0.5 ]
    xsum_Prev = gp.quicksum(self._x[i,j] for (i,j) in Prev)
    self.addConstr(xsum_Prev <= self._nsize-1)#配置場所が1つでも違えば別の解

if __name__ == '__main__':
  nsize = 10#11以上禁止。
  print("N-Queen Problem")
  for i in range(1,nsize+1):
    m = nQueenModel(tableSize = i) # i-Queen問題インスタンス
    solcnt = 0 #解の数カウンタ
    while m.checkFeas():#実行可能解が見つかる限り繰り返す
      m.inhibitPrevSol()#前回と同じ解を出さない制約
      solcnt += 1#カウンタをインクリメント
    print("Nsize = %2d, SolCount = %5d"%(i,solcnt))#実行可能解が見つからなくなった時点での解の数
