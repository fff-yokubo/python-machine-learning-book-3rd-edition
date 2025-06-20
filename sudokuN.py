

import sys
from sudokuSub import sudokuModel, read_ban



filename = sys.argv[1]

#盤面およびサイズ取得
ban = read_ban(filename)

#NOTE: 盤面データによる数独モデルインスタンス生成

# print(len(ban),len(ban[0]))


model = sudokuModel(ban,filename.replace(".csv",""))


#求解および結果表示
model.getResult()


print(model)