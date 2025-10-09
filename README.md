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
