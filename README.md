# 卒論
実行方法：
python3 main.py

出力：
output_data/output.png

アルゴリズム説明
①第一段階のパッキング
max. ∑w[i]*h[i] - a
s.t. w[i]*s[i] <= S[i] for all  group i (S[i] is constant)
     w[i] >= max{car_width} for i in group i
     h[i] >= max{car_height} for i in group i

②第二段階のパッキング
