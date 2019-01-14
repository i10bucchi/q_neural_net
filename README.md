# q_neural_net

## Overview

機械学習の勉強の一環で, QLearning+NeuralNetでゲームを学習するモデルをC言語で作成しました.

## Description

### Game

落ちてくるボールをバーで跳ね返すゲームです。
Demoをみてもらえるとわかりやすいと思います。

### Model

一般的なQLearningのルックアップテーブルをNeuralNetで推測します。
すなわち, NeuralNetは状態を入力としてとりQ値を出力する関数として使用します。
状態は以下の5つです。

- ボールのx座標
- ボールのy座標
- ボールのx方向速度
- ボールのy方向速度
- バーのx座標

行動は以下の3つです。

- バーを左へ
- バーを右へ
- バーを動かさない

つまりNeuralNetは入力層ニューロン5つ, 出力層ニューロン3つと中間層の構成になります。Demoは中間層1つでニューロン数は96に設定しています。

報酬は以下に沿って与えています。

- 跳ね返したら1.0
- 取り逃がしたら-1.0
- バーが端によっていなければ(0.01)

自分のところでは3つ目がないとずっと端に留まり続けていましました。

## Demo

<div align="center">
    <img src=./figure/demo1.gif>
    <p>図1. デモ</p>
</div>

## Requirement

ゲームの描画にHandy Graphicというものを使っています。
[こちら](http://www7a.biglobe.ne.jp/~ogihara/Hg/hg-jpn.html)からインストールして下さい.
また, 注意としてHandy GraphicはMacOSのみの対応となっています。

## Usage

```
# 以下で学習を行えます

# モデル(ニューラルネットの重み)を保存するフォルダをプログラムがあるフォルダ内に作成します
$ mkdir model
# コンパイル
$ hgcc learning_qnn.c helper.c MT.c -o learning
# 実行
$ ./learning

# 以下で学習したモデルでゲームを実行できます

# コンパイル
$ hgcc test.c helper.c MT.c -o game
# 実行
$ ./game
```

## Author

[takumi](https://github.com/i10bucchi)
