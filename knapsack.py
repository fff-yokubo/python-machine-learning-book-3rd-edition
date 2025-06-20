
# %%
import gurobipy as gp
import pandas as pd
import json

'''
データセット-1 8種類
'''
size =[21, 11, 15,  9, 34, 25, 41, 52]
val = [22, 12, 16, 10, 35, 26, 42, 53]

'''
データセット2
'''
# size =[21, 11, 15, 15, 34, 25, 41,  80] 
# val = [22, 12, 16, 10, 35, 26, 42, 200]
capa = 100

items = len(size)

m = gp.Model()
x = {k:m.addVar(vtype="B",name = "x[%s]"%(k+1)) for k in range(items)}

xsum_size = gp.quicksum(s * x[idx] for idx, s in enumerate(size))
m.addConstr(xsum_size <= capa, name = "Size" )

obj = gp.quicksum(v* x[idx] for idx, v in enumerate(val))
m.setObjective(obj, sense = gp.GRB.MAXIMIZE)

# xsumN = gp.quicksum(x.values())
# m.addConstr(xsumN >= 4, name = "Num" )

m.optimize()

if m.Status == gp.GRB.OPTIMAL:
  ("詰めたのは？")
  print("-"*50)

  size_sum = 0
  val_sum = 0
  for k, xv in x.items():
    if xv.X > 0.5:
      sk = size[k]
      vk = val[k]
      size_sum += sk
      val_sum += vk 
      print("Item %s, size %3s, value %3s"%(k,sk,vk))
  print("-"*50)
  print("Total: size %3s, value %3s"%(size_sum, val_sum))
else:
  m.computeIIS()
  m.write("knap.ilp")

m.write("knap.lp")
