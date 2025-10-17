# マルチタスクモデル学習・ビームサーチ運用ガイド

## 概要
本ガイドでは、`train_multitask.py`・`multitask_model.py`・`beam_search_runtime.py` の3つのスクリプトを組み合わせ、TLG8 ブロック選択を高速化するための学習〜推論〜エンコーダ連携方法をまとめる。学習が停滞・不安定な際の診断手順や対処方針も含める。

## 前提条件
- Python 3.9 以上
- `numpy`・`tqdm` など既存スクリプトが依存するライブラリ（`requirements.txt` が無い場合は `pip install numpy tqdm` 程度で十分）
- `tlgconv` バイナリ（`mkdir -p build && cmake --build build` で生成）
- 学習データ JSONL: 既存の学習ダンプ機構で得られる `pixels`・`best*`・`second*` を含む形式

## データ収集
`ann_training_workflow.md` と同様に `--tlg8-dump-training` 付きで `tlgconv` を実行し、1 行 1 ブロックの JSONL を蓄積する。

```bash
mkdir -p data
for img in test/images/*.bmp; do
    tag=$(basename "$img")
    build/tlgconv "$img" \
                  "/tmp/${tag%.bmp}.tlg8" \
                  --tlg-version=8 \
                  --tlg8-dump-training=data/tlg8_multitask.jsonl \
                  --tlg8-training-tag="$tag"
done
```

## 学習コマンド例
`train_multitask.py` は JSONL を読み込み、学習・評価・NPZ エクスポートまでを実行する。主な引数:

- `inputs`: JSONL ファイルまたはディレクトリ（複数指定可）
- `--epochs`: 既定 200。進捗を見つつ 300〜400 まで延長しても良い。
- `--lr`: 学習率。既定 `1e-3`。
- `--batch-size`: 既定 512。メモリに余裕があれば増やす。
- `--dropout`: トランク層のドロップアウト率。
- `--hidden-dims`: 共有トランクの層構成。
- `--epsilon-soft`: best/second のソフトターゲット比重（既定 0.2）。
- `--export-dir`: ベストモデルを書き出すディレクトリ。

実行例:

```bash
python3 train_multitask.py data/tlg8_multitask.jsonl \
    --epochs 240 \
    --lr 5e-4 \
    --batch-size 1024 \
    --dropout 0.1 \
    --hidden-dims 1536 768 384 \
    --weight-decay 3e-4 \
    --test-ratio 0.2 \
    --export-dir out/multitask
```

実行中はエポック毎に各ヘッド（predictor/filter_perm/filter_primary/filter_secondary/reorder/interleave）の `top1`・`top2`・`two_choice` が表示される。`--export-dir` を指定すると検証 `two_choice` が最良の時点で `multitask_best.npz` と `metrics.json` が保存される。

## 学習状態の評価指標
- **two_choice**: best または second を当てられた割合。目標 0.95 以上。
- **top1 vs top2 のギャップ**: `top1` が極端に低く `top2` が高い場合、2位情報に依存しすぎている。ビームサーチで救済できるが、ログを確認して極端なフェーズが無いか把握する。
- **損失推移**: `train_loss` と `val_loss` を比較し、後者が早期に下げ止まるようなら過学習兆候。

### 学習がうまくいっていない兆候と対処
1. **全フェーズで two_choice が 0.9 未満**
   - データ不足が第一候補。より多様なブロックを含む JSONL を追加し、再学習。
   - モデル容量を増やす: `--hidden-dims` に層を追加 or 各層を 1.5〜2 倍へ。
2. **特定のヘッドのみ低精度**（例: filter_primary）
   - `--epsilon-soft` を 0.3〜0.4 に増やし second 情報を強調。
   - データ内で second ラベルが欠落していないかサンプリングチェック。
3. **検証ロスが上昇し始める**
   - `--weight-decay` を 1e-4 → 3e-4 などへ強化。
   - `--dropout` を 0.15〜0.2 に増加。
   - 早期終了 (patience) の既定設定で止まらない場合は `--early-stop-patience` を短くする（オプションがある場合）。
4. **学習が不安定（loss が NaN）**
   - 学習率を半分 (`5e-4` → `2.5e-4`) に下げる。
   - JSONL に壊れた行が無いか `python3 train_multitask.py --dry-run` など読み込み専用モードでチェック（将来追加する場合）。

## 推論・ビームサーチの使い方
### モデル読み込みとトップ候補抽出
`multitask_model.py` の `MultiTaskModel.load` を利用する。特徴量は学習時と同じ前処理（8×8×C + 3 スカラー標準化）を行う。

```python
import numpy as np
from multitask_model import MultiTaskModel

model = MultiTaskModel.load("out/multitask/multitask_best.npz")
features = np.load("sample_block_features.npy")  # shape = (259,)
logits = model.predict_logits(features[None, :])
topk = model.predict_topk(features[None, :], k=2)
print(topk["head_predictor"])  # [(class_id, logprob), ...]
```

フィルタ候補は `build_filter_candidates` で 4 通りまでに束ねられる。

### ビームサーチによる構成選択
`beam_search_runtime.py` の `select_block_config` では、各フェーズの上位候補から最大 16 通りの組合せをエンコーダで評価する。

```python
from beam_search_runtime import select_block_config

# try_encode_block は (predictor, filter_code, reorder, interleave) を受け取りビット長を返す
cfg, bits = select_block_config(
    features,
    model,
    try_encode_block,
    margin_reorder=0.5,
    margin_interleave=0.5,
    margin_predictor=0.3,
)
print(cfg, bits)
```

フィルタの合成順位と候補圧縮は自動で行われ、信頼度マージンにより組合せ上限が 16 を超えないよう制御される。`try_encode_block` は既存のエンコーダ API に合わせて実装する。

### テスト
最低限のユニットテストは `python3 -m unittest discover -s test -p 'test*.py'` で実行できる。新しいモデル構成を導入する際は成功することを確認してからコミットする。

## トラブルシューティング
| 症状 | 原因候補 | 対処 |
| --- | --- | --- |
| `predict_topk` の結果が常に同じ ID | 標準化ミス（mean/std 適用漏れ） | `MultiTaskModel.load` の `feature_scaler` を利用し、推論時も同じ手順で正規化する |
| ビームサーチで 16 超の試行が発生 | 自前修正で制御が外れた | `max_trials` の assert を再確認し、マージン値を適切に設定する |
| try_encode_block が重く推論が遅い | 逐次 `tlgconv` 呼び出しになっている | C++ バインディング内で直接呼び出す / Python からはバッチ化は困難なのでログキャッシュを検討 |
| NPZ 読み込みで `KeyError: 'mean'` | エクスポート結果が古い形式 | `train_multitask.py` で再学習し最新フォーマットへ更新 |

## 運用 Tips
- 学習ログは標準出力だけでなく `--log-file` を追加して保存すると、収束傾向を後から確認しやすい（オプションが無い場合は `tee` を活用）。
- 検証 two_choice が 0.97 以上で安定したら、本番エンコーダに組み込み、ログで試行回数・平均ビット数を観測すると効果が把握できる。
- 追加データが手に入ったら既存 NPZ を初期値としてロードし、微調整 (fine-tuning) することも検討できる。`train_multitask.py --load` のようなオプションを将来的に追加する際の叩き台になる。

以上でマルチタスク学習スクリプトとビームサーチ実行環境の基本運用を網羅する。
