# %%
import gurobipy as gp
import numpy as np

import pprint as pp



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

      # ban = np.array(ban)
      
      ny = len(ban)
      nx = len(ban[0])
      
      if (nx == ny ) and (nx in (4,9,16,25)):
        return ban
      else:
        print("盤面データ不正")      
      


  except:
    print("盤面ファイル: %s 読み込みエラー"%filename)


#盤面およびサイズ取得
ban = read_ban("test.csv")

#盤面サイズ: 必ず平方数(4x4 or 9x9, 16x16)
nsize = len(ban)#セクションの数 サイズの平方根(4x4⇢2, 9x9 ⇢3, 16x16⇢n) 
nsect = int(np.sqrt(nsize))


model = gp.Model('%s x %s Sudoku'%(nsize,nsize))


#決定変数 盤面
x = {
  (i,j,n): model.addVar(vtype = gp.GRB.BINARY, name = "x[(%s,%s):%s]"%(i,j,n+1))
  for i in range(nsize) for j in range(nsize) for n in range(nsize)
}


model.update()

#NOTE: 成約1: 盤面各マスには1からnsizeの数値いずれか1つしか割り当てない

for i in range(nsize):
  for j in range(nsize):
    xsumIJ = gp.quicksum(x[i,j,n] for n in range(nsize))
    model.addConstr(xsumIJ == 1, name = "Constr1[%s,%s]"%(i,j))

#NOTE: 制約2I/J: 1からnsizeの数値は、盤面縦(I)/横方向(J)で1回ずつしか登場しない
  for n in range(nsize):
    xsumJ = gp.quicksum(x[i,j,n] for j in range(nsize))
    model.addConstr(xsumJ ==1, name ="Constr2I[%s:%s]"%(i,n+1))

    xsumI = gp.quicksum(x[j,i,n] for j in range(nsize))
    model.addConstr(xsumI ==1, name ="Constr2J[%s:%s]"%(i,n+1))


#NOTE: 制約3 セクションで各数値は1回しか登場しないこと
for sx in range(nsect):
  for sy in range(nsect):
    for n in range(nsize):
      xsumSect = gp.quicksum( x[i,j,n]
        for j in range(nsize) if j // nsect == sx
        for i in range(nsize) if i // nsect == sy
      )
      model.addConstr(xsumSect == 1, name = "Constr3[(%s,%s):%s]"%(sx,sy,n+1))

#NOTE: 制約4-初期盤面データの読み込み
for i, bani in enumerate(ban):
  for j,banij in enumerate(bani): 
    if banij > 0:
      model.addConstr(x[i, j , banij-1]==1, name ="Constr4[(%s,%s)]=%d"%(i,j,banij))


import random

obj = gp.quicksum(random.random()*xv for xv in x.values())

model.setObjective(obj, sense = gp.GRB.MINIMIZE)


model.optimize()

if model.Status == gp.GRB.OPTIMAL:
  result = []
  for i in range(nsize):
    resulti = []
    for j in range(nsize):
      for n in range(nsize):
        if x[i,j,n].X>0.5:
          resulti.append(n+1)
          break
    result.append(resulti)

  print("Input:")
  pp.pprint(ban)

  print("Result:")
  pp.pprint(result)

# %%
