# 20日間のローソク足画像からCNNで株価の上下を予想するモデル
- 参考論文: https://arxiv.org/pdf/1903.12258.pdf
	- 論文では上がり下がりを予測するバイナリラベルでvalidationの正解率>0.9
	- VGG16より小さいモデルで結果出してた（学習率などのハイパラは書いてない不親切な論文だった。validationとtestを同じデータでやってるし）
- 株価データベースと株価csvが必要
	- https://github.com/riron1206/03.stock_repo/tree/master/sqlite_analysis
- 02_keras_pyライブラリも必要
	- https://github.com/riron1206/02_keras_py

## ラベルの分け方かえていくつかモデル作ったが、以下のラベルを採用した
- 予測実行日の1日後の始値 > 2日後の始値なら「0」
- 予測実行日の1日後の始値 < 2日後の始値なら「1」
- 予測実行日の1日後の始値 * 1.05 < 2日後の始値なら「2」
- ※上がり下がりを予測するバイナリラベルだとvalidationの正解率>0.9になるが、最近のデータで試すとあまり当たらない感じだったのでこうした（validation setの切り方が悪いためtest setの分布と合ってないのか？）
- 予測実行日の終値 < 1日後の始値 and 予測ラベル=1or2 and 確信度>0.8 のデータは良さそう

![CM_without_normalize_optuna_best_trial_accuracy.png](https://github.com/riron1206/candlestick_model/blob/master/CM_without_normalize_Xception_2day_label.png)

## 行った手順
#### 1. notebook/*.ipynb でデータ作成、モデル作成試す
#### 2. 時系列の分け方でデータセット作成
- train setは2000-2017年まで、test setは2018-2020年までのデータを使う
```bash
# 画像作成
$ python make_candlestick.py --is_mav_png --is_2day_label -o D:\work\candlestick_model\output\orig_image_all_2day_label
※画像200万枚ぐらいつくるため48時間近くかかる

# train/test setに分割
$ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_2day_label -o D:\work\candlestick_model\output\ts_dataset_all_2day_label --is_test_only --is_not_equalize
```
#### 3. code/tf_base_class*.py でモデル作成（パラメータチューニングも可能。少量データ版では試した）
```bash
$ python tf_base_class_all_data_2day_label.py -m train
```
#### 4. bestモデルで予測
```bash
# JPX日経インデックス400 + 日経225 + 日経500種について
$ python tf_predict_best_model.py --is_2day_label

# 7269:スズキについて数日さかのぼって実行。下記は10日さかのぼる（2020/07/31からさかのぼって10日間毎日予測）
$ python tf_predict_best_model.py -c 7269 -d 2020-07-31 -t_d 10 --is_2day_label
```
