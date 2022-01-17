## 定式化
### 第一段階のパッキング  
変数：x[i],y[i],w[i],h[i] (for all i)   
入力：segment-size(I), S[i]  
<br>
max. ∑ w[i]*h[i]<br>
s.t. <br> 
+ w[i]*s[i] <= S[i] (for all  group i)<br>
+ w[i] >= max{car_width} (for i in group I)<br>  
+ h[i] >= max{car_height} (for i in group I)  <br>
+ ∑w[i]*h[i](for i in group I) <= segment-size(I)
+ 各長方形の位置関係<br>


### 第二段階のパッキング
変数：x[j], y[j]<br>
入力：w[j], h[j], X[i], Y[i], W[i], H[i]
<br>
<br>
目的関数なし<br>
s.t. <br>
+ X[i] < x[j] < X[i] + W[i] (car[j] is in group i)<br>
+ Y[i] < y[j] < Y[i] + H[i] (car[j] is in group i) <br>
+ レクトリニアでの駐車に関する制約