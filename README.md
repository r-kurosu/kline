## 実行方法
```python3 main.py```

必要であれば以下のマクロ変数を編集．\
L20: ```BOOKING``` = 1, 2, 3 \
L21: ```COLOR``` = 6, 7 
(6: LP, 7: DP)

## 出力
output_data/output.png

## tex
paperディレクトリに移動 \
```latexmk template.tex```

## その他ファイル説明
data/ : main.pyで使うデータファイル \
fill_rate.py: 上界下界の計算用 \
test/make_csv.py: main.pyで使うデータを作るためのプログラム
