# DINOv3 Interactive Feature Visualization Demo

Google Colab上で動作するDINOv3の特徴量可視化デモです。画像をアップロードして、マウス操作で特徴量を探索し、疑似的なセグメンテーションを行うことができます。

## 機能

- 📸 **画像アップロード**: 任意の画像をアップロードして解析
- 🖱️ **インタラクティブ探索**: マウスホバーで各ピクセルの特徴量情報を表示
- 📍 **基準点選択**: クリックで基準点を固定
- 🎨 **セグメンテーション**: 基準点との特徴量類似度に基づく疑似セグメンテーション
- 📊 **統計情報**: 特徴量の統計情報をリアルタイム表示

## Google Colabでの使用方法

### 1. 新しいColabノートブックを作成

[Google Colab](https://colab.research.google.com/)で新しいノートブックを作成します。

### 2. 必要なライブラリをインストール

```python
!pip install torch torchvision matplotlib opencv-python pillow ipywidgets timm
```

### 3. デモコードを実行

以下のコードをセルにコピー&ペーストして実行してください：

```python
# GitHubからコードをダウンロード
!wget https://raw.githubusercontent.com/[YOUR_USERNAME]/dinov3-demo/main/dinov3_interactive_demo.py

# デモを実行
exec(open('dinov3_interactive_demo.py').read())
```

または、コード全体を直接コピー&ペーストすることも可能です。

### 4. 使用手順

1. **画像アップロード**: 「画像をアップロード」ボタンをクリックして画像を選択
2. **特徴量探索**: 画像上でマウスを動かすと、各位置の特徴量情報が表示されます
3. **基準点選択**: 興味のある領域をクリックして基準点を設定
4. **セグメンテーション**: 「セグメンテーション実行」ボタンをクリックして類似領域を可視化

## 技術詳細

### DINOv3について

DINOv3（DINO version 3）は、Metaが開発した自己教師あり学習による視覚変換器（Vision Transformer）です。ラベルなしの画像データから豊富な視覚表現を学習し、様々な下流タスクで優秀な性能を発揮します。

### 特徴量抽出

- **モデル**: DINOv3-Small (ViT-S/14)
- **入力サイズ**: 224×224ピクセル
- **パッチサイズ**: 14×14ピクセル
- **特徴量次元**: 384次元

### 類似度計算

基準点の特徴量と全ピクセルの特徴量間でコサイン類似度を計算し、類似度マップを生成します。

## ファイル構成

```
dinov3-demo/
├── dinov3_interactive_demo.py  # メインデモコード
├── requirements.txt            # 必要なライブラリ
├── README.md                  # このファイル
└── examples/                  # サンプル画像（オプション）
```

## 動作環境

- Python 3.7+
- Google Colab（推奨）
- GPU利用可能（CPUでも動作しますが、処理が遅くなります）

## トラブルシューティング

### モデル読み込みエラー

```python
# 代替方法でモデルを読み込む場合
import timm
model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
```

### メモリ不足

大きな画像を使用する場合は、事前にリサイズしてください：

```python
from PIL import Image
image = Image.open('your_image.jpg')
image = image.resize((224, 224), Image.LANCZOS)
```

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 参考文献

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Facebook Research DINOv2](https://github.com/facebookresearch/dinov2)
