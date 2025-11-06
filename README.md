# tlg8

## ビルド方法

Release ビルドは次のコマンドで行えます。

```
./tools/pgo_build.sh release
```

## PGO ワークフロー

テスト画像 (`test/images/*`) を使った PGO (Profile-Guided Optimization) の実行手順は以下の通りです。

1. プロファイルを生成します。

   ```
   ./tools/pgo_build.sh pgo-generate
   ```

2. 生成されたプロファイルを用いて最適化済みバイナリをビルドします。

   ```
   ./tools/pgo_build.sh pgo-use
   ```

不要になった PGO 用ビルドディレクトリとプロファイルは次で削除できます。

```
./tools/pgo_build.sh clean
```

## reorder ヘッドスイープ (v2)

`tools/sweep_reorder.py` を使うと reorder ヘッド向けのハイパーパラメータスイープを簡単に実行できます。結果は `runs/sweeps/reorder_<TAG>/` 以下にログ (`logs/*.log`)、中間成果物、`results_v2.csv` / `results_v2.md` として保存されます。

- `--tag` : サブフォルダ名に利用するタグ (既定: `v2`)
- `--grid KEY=v1,v2,...` : 指定したパラメータのみグリッドを上書き
- `--concurrency` : 同時実行ジョブ数。GPU が複数ある場合は `--cuda-devices` で割り当て可能
- `--resume` : 既存の `results_v2.csv` を読み込み、完了済み (`status` in {`ok`,`oom`}) の組み合わせをスキップ
- `--max-runs` : デバッグ用に実行件数を制限
- `--dry-run` : コマンド確認のみで実行しない

OOM が発生した場合は自動的に次に小さいバッチサイズへ縮小して再試行します。最終結果は `status={ok,oom,fail}` として CSV に記録され、Markdown では `val_reorder_top1` でソートした Top10 と再現コマンドが生成されます。

実行例:

```
python3 tools/sweep_reorder.py \
  --tag v2 \
  --grid beta=0.9999,0.995 alpha=0.5,1.0 gamma=1.8,2.5 eps=0.00,0.05 lr=3e-5 batch=32768 sched=cosine_warm5 \
  --max-runs 24 \
  --concurrency 2
```
