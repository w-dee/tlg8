# TLG8 ANN 学習ワークフロー（ローカル環境向け）

## ゴール
- TLG8 エンコーダーがブロック単位で選んだ最適な処理フロー（predictor/filter/reorder/interleave/entropy）を、画素データのみから推論する ANN を Python 上で学習する。
- サイド情報は入力特徴として使わず、推論対象そのものとして 95% 以上の正答率（top1+second 指標）を達成する。
- 学習〜評価までを著作権保護されたローカル画像資産だけで再現できる手順を整備する。

## 前提準備
1. **ビルド**  
   ```bash
   mkdir -p build && cmake --build build
   ```
2. **依存ライブラリ**  
   Python 3.9 以降＋`numpy` が必要。インストール例:
   ```bash
   python3 -m pip install --user numpy
   ```

## 学習データの収集
1. 収集結果を書き出す JSONL ファイル（例: `data/tlg8_training.jsonl`）を用意する。
2. `tlgconv` に学習ダンプ用オプションを付けてエンコードを実行する。例として `test/images` にある BMP をまとめて処理する場合:
   ```bash
   mkdir -p data
   for img in test/images/*.bmp; do
       tag=$(basename "$img")
       build/tlgconv "$img" \
                     "/tmp/${tag%.bmp}.tlg8" \
                     --tlg-version=8 \
                     --tlg8-dump-training=data/tlg8_training.jsonl \
                     --tlg8-training-tag="$tag"
   done
   ```
   - `--tlg8-dump-training` : 追記先の JSONL ファイル。
   - `--tlg8-training-tag` : 画像識別子（任意文字列）。タグはモデルの入力には含めない。
   - 出力ファイルは `/tmp` など任意の一時パスでよい（生成された `.tlg8` を削除しても学習データは残る）。
3. 手元の大規模画像コーパスを使う際も同じコマンドを、ローカルディレクトリに対して実行する。著作権的にクラウドへアップロードする必要はなく、JSONL もローカルに留まる。

## モデル学習
### 推奨設定
- 多層パーセプトロン (MLP) モデルを使用。
- 標準化あり（既定で有効）。
- ミニバッチ 1024、エポック 400、学習率 5e-4、ドロップアウト 0.2 でスタート。

実行例:
```bash
python3 tools/train_tlg8_ann.py data/tlg8_training.jsonl \
    --model mlp \
    --hidden-dims 1536 768 384 \
    --dropout 0.2 \
    --batch-size 1024 \
    --epochs 400 \
    --learning-rate 5e-4 \
    --weight-decay 1e-4 \
    --test-ratio 0.1 \
    --export-dir models/tlg8_ann
```
- エポックごとに訓練・評価精度（top1 / top2 / top1+second）を表示し、改善が止まると自動で早期終了する。
- `--export-dir` を指定すると各フェーズの重み (`*_model.npz`) と `metrics.json` が保存され、Python 上の推論に再利用できる。

### ハイパーパラメータ調整のヒント
- **正答率が伸びない場合**: 隠れ層次元を増やす（例: `--hidden-dims 2048 1024 512 256`）、エポック数を 600〜800 に伸ばす、ドロップアウトを 0.1〜0.3 の範囲で調整する。
- **過学習気味の場合**: `--test-ratio` を 0.2 へ上げて評価サンプルを増やす、`--weight-decay` を 3e-4 などへ引き上げる。
- **データが多い場合**: `--batch-size` を 2048 などへ増やし学習を高速化する。十分な GPU が無い環境でも CPU + numpy だけで動作する。

## 学習結果の確認
1. コマンド出力の表形式サマリーで各フェーズの top1 / top2 / top1+second を確認する。
2. `metrics.json` には小数（0-1）で同じメトリクスが保存される。例:
   ```json
   {
     "predictor": {
       "train_top1": 0.982,
       "train_top2": 0.996,
       "train_two_choice": 0.998,
       "test_top1": 0.951,
       "test_top2": 0.992,
       "test_two_choice": 0.996
     },
     ...
   }
   ```
3. 95% 到達判定は `test_two_choice` (top1+second) を基準にする。全フェーズで 0.95 以上を目標とし、達しないフェーズがあればそのフェーズのハイパーパラメータを重点的に再調整する。

## 推論テスト（Python）
エクスポートした `*_model.npz` は numpy の `np.load` で読み込み、`mean`/`std` を用いて標準化した特徴量に対し `tools/train_tlg8_ann.py` 内の `model_logits_from_state` を流用すれば推論できる。例:
```python
import json
import numpy as np
from tools.train_tlg8_ann import apply_feature_scaler, model_logits_from_state

checkpoint = np.load("models/tlg8_ann/predictor_model.npz")
state = {k: checkpoint[k] for k in checkpoint.files}
mean, std = state.pop("mean"), state.pop("std")
feature = ...  # JSONL で使用したのと同じ 259 要素のベクトル
x = apply_feature_scaler(feature[None, :].astype(np.float64), mean, std)
logits = model_logits_from_state(state, x)
top3 = np.argsort(logits[0])[::-1][:3]
print(top3)
```
この方法で Python 内でサイド情報推定のオフライン検証が可能になる。

## 95% 達成への実運用ステップ
1. 画像コーパスを数千〜数万枚規模で収集する（多様なコンテンツを含める）。
2. 上記のバッチスクリプトで JSONL を生成し、`data/` 以下に蓄積する。
3. 推奨設定で学習 → 指標を確認 → 足りないフェーズだけ再学習、と段階的に改善する。
4. 安定した指標が得られたら `models/` 以下の NPZ を保存・共有し、C++ 統合フェーズまで Python での推論検証を継続する。

以上の手順に従えば、サイド情報を入力に含めずとも高精度推論をローカル環境だけで再現できる。
