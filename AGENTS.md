##  natural language usage

 - 日本語で回答してください
 - ソースコード中のコメントは日本語

## about building

 - ビルドコマンドは `mkdir -p build && cmake --build build`


## about testing

 - テスト画像に対してエンコード→デコードを行い、元の画像と一致するか(一致すれば成功)をテストするラウンドトリップテストは → `python test/run_roundtrip.py --tlgconv build/tlgconv --images test/images`  このテストは数分かかります。気長に待ってね。