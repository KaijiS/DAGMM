# DAGMM

## 概要
[DAGMM](https://openreview.net/pdf?id=BJJLHbb0-)をchainerを用いてインプリ  

例としてArrhythmiaデータセットを用いた．  

今回は正常データを正常として学習(上記論文中では正常以外ラベルのうちのいくつかのラベルも正常とみなして学習した)．  
そのほかハイパーパラメータは致させている．

## 実行方法

### 学習
学習させるコマンドは以下である．  
`python3 train.py`  

GPUで演算する場合は引数を用意  
`python3 train.py -g [デバイス番号>=0]`

その他の引数の種類は  
`python3 train.py -h`  
で確認できる．

主にいじると思われる引数は，  
- `-g` GPU使用するならデバイス番号 
- `-e` 何回学習するかのepoch数  
- `-f` 何epoch毎に学習モデルを保存するか  
- `-r` 保存されたモデルから学習を再開するなら何epochから開始か

くらいだと思われる．  
他はハイパーパラメータなどの設定である．

デフォルトでは学習モデル(途中も)やlog，誤差の推移グラフなどが `./result` ディレクトリの中に入るようになっている．(ディレクトリがない場合は作成, ディレクトリ名の変更は引数で `-o ディレクトリ名` と指定すれば良い)


### テスト
論文中ではテスト時にenergyが大きいほど異常と推定するようであるが，energy以外に尤度を表すgammaや，Autoencoder(Compression Network)で再構築した時の誤差をヒストグラムで色別(正常(学習，検証)，異常)で可視化してみた．さらにAutoencoder(Compression Network)の潜在変数zも可視化してみた．それぞれ実行するコマンドは以下である．

`python3 test_energy.py`  
`python3 test_gamma.py`  
`python3 test_reconstruct.py`  
`python3 test_z.py`  

これらも引数でパラメータを変更できる．
主にいじると思われる引数は  
`-e` 何回学習されたモデル(何epoch学習したモデル)を使用するか  
であると思われる．
