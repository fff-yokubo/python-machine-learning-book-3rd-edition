import gurobipy as gp
import pprint as pp
import sys

import numpy as np
class sudokuModel(gp.Model):
  
  def __init__(self, ban,filename):

    '''
    N x N 数独モデル生成
    '''
    
    #盤面サイズ: 必ず平方数(4x4 or 9x9, 16x16)
    self._ban = ban
    self._nsize = len(ban)#セクションの数 サイズの平方根(4x4⇢2, 9x9 ⇢3, 16x16⇢n) 
    self._nsect = int(np.sqrt(self._nsize))
    self._filename = filename
    name = 'Sodoku InputData: %s, size= %s x %s'%(filename, self._nsize, self._nsize)
    super().__init__(name)

    # #決定変数 盤面
    self._x = {
      (i,j,n): self.addVar(vtype = gp.GRB.BINARY, name = "x[(%s,%s);%s]"%(i,j,n+1))
      for i in range(self._nsize) for j in range(self._nsize) for n in range(self._nsize)
    }

    self.update()

    #NOTE: 制約1: 盤面各マスには1からself._nsizeの数値いずれか1つしか割り当てない

    for i in range(self._nsize):
      for j in range(self._nsize):
        xsumIJ = gp.quicksum(self._x[i,j,n] for n in range(self._nsize))
        self.addConstr(xsumIJ == 1, name = "Constr1[(%s,%s)]"%(i,j))

    #NOTE: 制約2I/J: 1からself._nsizeの数値は、盤面縦(I)/横方向(J)で1回ずつしか登場しない
      for n in range(self._nsize):
        xsumJ = gp.quicksum(self._x[i,j,n] for j in range(self._nsize))
        self.addConstr(xsumJ ==1, name ="Constr2I[%s;%s]"%(i,n+1))

        xsumI = gp.quicksum(self._x[j,i,n] for j in range(self._nsize))
        self.addConstr(xsumI ==1, name ="Constr2J[%s;%s]"%(i,n+1))

    #NOTE: 制約3 セクションで各数値は1回しか登場しないこと
    for sx in range(self._nsect):
      for sy in range(self._nsect):
        for n in range(self._nsize):
          xsumSect = gp.quicksum( self._x[i,j,n]
            for j in range(self._nsize) if j // self._nsect == sx
            for i in range(self._nsize) if i // self._nsect == sy
          )
          self.addConstr(xsumSect == 1, name = "Constr3[(%s,%s);%s]"%(sx,sy,n+1))

    #NOTE: 制約4-初期盤面データの読み込み
    for i, bani in enumerate(ban):
      for j,banij in enumerate(bani): 
        if banij > 0:
          self.addConstr(self._x[i, j , banij-1]==1, name ="Constr4[(%s,%s)]=%d"%(i,j,banij))

    lpfile = "%s.lp"%self._filename
    self.write(lpfile)
    print("End: Defining %sx%s Sudoku Model %s"%(self._nsize, self._nsize,lpfile))


  def getResult(self):

    '''
    求解および結果抽出
    生成したモデルを求解
      解ありの場合⇢盤面結果を取り出し
      解なし(INFEASIBLE)の場合⇢
    '''
  
    self.optimize()
    
    if self.Status != gp.GRB.INFEASIBLE:
      result = []
      for i in range(self._nsize):
        resulti = []
        for j in range(self._nsize):
          for n in range(self._nsize):
            if self._x[i,j,n].X>0.5:
              resulti.append(n+1)
              break
        result.append(resulti)

      print("\nInput: %s"%filename)
      pp.pprint(ban)

      print("\nResult:")
      pp.pprint(result)
      print("\n")
      # pp.pprint(self._ban)

    else:
      self.computeIIS()
      ilpfile= "%s_IIS.ilp"%self._filename
      self.write(ilpfile)

      print("Model INFEASIBLE: Check IIS file %s"%ilpfile)


def read_ban(filename):
  '''
  盤面読み込み
  http://www10.plala.or.jp/t_itokin/
  でDL可能な盤面データを読み込みしてnparray二次元配列に変換
  %%% 0は空欄とする
  '''

  try:
    ban = []
    with open(filename) as f:
      lines = f.read()
      for l in lines.split("\n"):
        str = l.split(",")#カンマ区切りを文字列配列に変換
        ls = [int(s) if s != '' else 0 for s in str ]#文字列配列を数値配列に変換
        ban.append(ls)
      ny = len(ban)
      nx = len(ban[0])
      
      #盤面が正方形　かつ、サイズが平方数でない場合⇢エラーを返す
      if (nx == ny ) and (nx in (4,9,16,25)):
        return ban
      else:
        print("盤面データ不正")      
      
  except:
    print("盤面ファイル: %s 読み込みエラー"%filename)
    
    


if __name__ == "__main__":

  filename = sys.argv[1]

  #盤面およびサイズ取得
  ban = read_ban(filename)

  #NOTE: 盤面データによる数独モデルインスタンス生成
  model = sudokuModel(ban,filename.replace(".csv",""))


  #求解および結果表示
  model.getResult()


  print(model)
