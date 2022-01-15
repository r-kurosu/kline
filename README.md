# 実行方法：
python3 main.py

# 出力：
output_data/output.png

# 定式化
1. 第一段階のパッキング
変数：x[i],y[i],w[i],h[i] (for all i) 
入力：segment-size, S[i]

max. ∑ w[i]*h[i]
s.t. w[i]*s[i] <= S[i] (for all  group i)
     w[i] >= max{car_width} (for i in group i)
     h[i] >= max{car_height} (for i in group i)
     + 各長方形の位置関係


2. 第二段階のパッキング
変数：x[j], y[j]
入力：w[j], h[j], X[i], Y[i], W[i], H[i]

s.t. X[i] < x[j] < X[i] + W[i] (car[j] is in group i)
     Y[i] < y[j] < Y[i] + H[i] (car[j] is in group i)
     + レクトリニアでの駐車に関する制約