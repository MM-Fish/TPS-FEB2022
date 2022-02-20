# playgound-series-feb-2022

## コンペ概要
- 微生物の種類を分類するコンペ
https://www.kaggle.com/c/tabular-playground-series-feb-2022

- 参考論文
https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full
迅速なDNAを用いた種同定方法の開発
そのため、データに意図的にバイアスなどを加えて、精度の高い種同定が出来るかどうかを調査している。
trainとtestで各菌株で1個体ずつのゲノムを使用(NCBIrefから取得)
1ゲノムから1000サンプル作成可能
→論文上では10×1000=10,000サンプルを使用

### データ
列：塩基ヒストグラムーATGCの4種類の塩基から組み合わせありで10個取得するパターン（合計286パターン）
行：サンプル
各列の値：(塩基ヒストグラムのスペクトル) - (完全にランダムなバイアス) → 2/19のnote確認
バイアススペクトル = 1/4^k × ( k! / (w!x!y!z!) ) → k=10
完全にランダムなバイアススペクトルを取り除くことで、種特異の値を検出しやすくする。
A10T0G0C0より、A4T3G3C3の方が生じやすい。→前者は順序を加味しても1通りだが、後者は 10!/(4!3!3!3!) = 700通り存在する
このようなバイアスを取り除くために、完全にランダムなバイアスで減算する。

- データ作成方法
①1個体のゲノムから？個のゲノム(仮想の1000個体)を作成
②各ゲノムから塩基ヒストグラムの確率分布(各個体が一意にもつ塩基ヒストグラムの確率)を作成 
③r個のBOC(10個の塩基、A2T3G3C2など)をランダムに取得して塩基ヒストグラム（実験により作成されるヒストグラム）を作成する。
④エラー率mの二項分布に従って、塩基ヒストグラムに誤差や突然変異を加える。
→ ③において、r個のうち、通常は②からサンプリングするが、エラーの時には完全ランダムバイアスからサンプリングする。
⑤作成された塩基ヒストグラムから（完全にランダムなバイアス）を減算する。

## 22/02/09
データをそのままでLightGBMで予測


## 22/02/15
kerasの実装
データをそのままでkerasを使って予測

## 22/02/19
### ラマン分光について調査
ラマン分光法とは、この入射光と異なった波長をもつ光（ラマン散乱光）の性質を調べることにより、物質の分子構造や結晶構造などを知る手法
- スペクトル
分光した結果を波長の強さなどで表した図

- ラマンスペクトル
[ラマン効果についてもっと知りたい方にへ（中段の図）](https://www.nanophoton.jp/lecture-room/raman-spectroscopy/lesson-1-1)が参考になる
分子が外部から光学エネルギーを与えられると、光電場の振動に応じて誘起双極子モーメント（分子？）が振動することで、照射した光と同じ振動数の散乱光が生じます（レイリー散乱光）
一個一個の分子を見てみると、それら自身も一定の周期で振動しています（固有振動）
光の電場の振動（振動数：νⅰ）と固有振動（振動数：νc）の干渉により散乱光が生じる（ラマン散乱光）

- ラマン分光の数字の見方
横軸がラマンシフト->[励起レーザ波長とラマンシフトの関係](https://www.horiba.com/jp/scientific/products-jp/raman-spectroscopy/about-raman/1/)
縦軸がラマン強度（物質の濃度などが分かる）

### 今回のデータ
```
Each row of data contains a spectrum of histograms generated by repeated measurements of a sample, each row containing the output of all 286 histogram possibilities (e.g., A0T0G0C10 to A10T0G0C0), which then has a bias spectrum (of totally random ATGC) subtracted from the results.
```
各行のデータは塩基ヒストグラムのスペクトル（ラマン強度）← ラマン分光系により1つのサンプルを繰り返し測定したものも含む
測定結果から完全にランダムなバイアススペクトルが引かれている？


```
The data (both train and test) also contains simulated measurement errors (of varying rates) for many of the samples, which makes the problem more challenging
```
多くのサンプルで測定誤差を含んでいる（コンペ作成者の意図的に？）

- point
ランダム誤差と測定誤差を上手く取り除いて、学習させると高いスコアが得られそう


### idea
- [論文に記載してある](https://www.nature.com/articles/s41467-019-12898-9#Sec7)
pcaからのlrとsvmを試してみる

- アンサンブル(knnとlgb)

- 読む
https://www.kaggle.com/odins0n/tps-feb-22-eda-modelling/
https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full
→この論文をもとに実装すればおそらく99%の精度を超える？

- 全部カテゴリカル変数に変える
ノイズが入ってしまい厄介なので、平均より高いor低いのカテゴリカル変数に変える？


## 22/02/20
- 論文を読んでデータの作成方法を理解した
