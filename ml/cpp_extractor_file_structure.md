# C++ エクストラクター出力ファイル仕様（tlgconv / TLG8）

このドキュメントは、`tlgconv` の TLG8 エンコード時に出力できる以下 3 種類のファイルについて、**実装（`src/tlg8_encode.cpp`, `src/label_record.h`, `src/tlg8_io.cpp`）に基づいた**形式仕様と、Python で読むときの実装例をまとめたものです。

- `--tlg8-dump-training=<path>`：学習用 JSONL（1 行 1 ブロック）
- `--label-cache-bin=<path>`：ラベルキャッシュ（固定長バイナリ）
- `--label-cache-meta=<path>`：ラベルキャッシュのメタデータ（JSON）

## 共通：ブロック走査順（重要）

`--tlg8-dump-training` の JSONL も、`--label-cache-bin` のレコード列も、**同じブロック走査順**で出力されます。

走査順は以下です（エンコーダ内部の順序）：

1. 画像をタイルに分割して、`origin_y` 昇順 → `origin_x` 昇順で処理
   - タイルサイズは固定：`tile_w=8192`, `tile_h=80`
2. タイル内を 8×8 ブロック単位で、`block_y` 昇順 → `block_x` 昇順で処理
3. 画像右端／下端では、ブロックサイズが 8 未満になることがあります（`block_size=[block_w, block_h]` がそれを表します）

したがって、**同一実行で JSONL と label-cache を両方出力した場合**、`i` 番目の JSONL 行と `i` 番目の `LabelRecord` は同じブロックに対応します（ただし label-cache 単体では座標情報を持ちません）。

---

## 1. `--tlg8-dump-training`（JSONL）

### 概要

- 追記モード（`ab`）で書き込みます。既存ファイルがあると末尾に追加されます。
- **1 行 = 1 ブロック**の JSON オブジェクト（JSON Lines / JSONL）です。
- 文字列は JSON のエスケープ規則に従い、ファイル自体は実質 UTF-8 として扱えます（C++ 側はバイト列として出力）。

### 1 行の JSON スキーマ

各行は概ね次のキーを持ちます（出力順は固定ですが、JSON としては順序非依存です）：

- `image`：`--tlg8-training-tag` で与えた文字列（未指定なら空文字列）
- `image_size`：`[image_width, image_height]`（どちらも整数）
- `tile_origin`：`[origin_x, origin_y]`（整数、タイル左上）
- `block_origin`：`[x, y]`（整数、ブロック左上＝`tile_origin + (block_x, block_y)`）
- `block_size`：`[block_w, block_h]`（整数、通常 8×8 だが端では小さくなる）
- `components`：色成分数（整数、3 または 4）
- `pixels`：ブロック画素のフラット配列（0–255 の整数）
  - 並び：`y` → `x` → `comp`（行優先、ピクセル内で成分が最後）
  - `components==3` のとき：`[R,G,B, R,G,B, ...]`
  - `components==4` のとき：**ARGB 順**で `[A,R,G,B, A,R,G,B, ...]`
  - 長さ：`block_w * block_h * components`
- `best`：最良候補（オブジェクト）または `null`
- `second`：次点候補（オブジェクト）または `null`

`best` / `second` の候補オブジェクトは以下のキーを持ちます：

- `predictor`：予測器インデックス（0–7）
- `filter`：カラー相関フィルターコード（0–95、`components>=3` の場合）
- `reorder`：並び替え方式インデックス（0–7）
- `interleave`：インターリーブ方式（0:None, 1:Interleave）
- `entropy`：エントロピー方式（0:Plain, 1:RunLength）
- `bits`：推定ビット長（整数、`uint64` 相当）

`reorder` の 0–7 は実装上、以下の名前に対応します：

```
0: hilbert
1: zigzag_diag
2: zigzag_antidiag
3: zigzag_horz
4: zigzag_vert
5: zigzag_nne_ssw
6: zigzag_nee_sww
7: zigzag_nww_see
```

### Python で読む例（ストリーミング）

巨大になりやすいので、全件 `json.load()` ではなく **1 行ずつ処理**するのが無難です。

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class Candidate:
    predictor: int
    filter: int
    reorder: int
    interleave: int
    entropy: int
    bits: int


@dataclass(frozen=True)
class TrainingRow:
    image: str
    image_size: tuple[int, int]
    tile_origin: tuple[int, int]
    block_origin: tuple[int, int]
    block_size: tuple[int, int]
    components: int
    pixels: list[int]
    best: Optional[Candidate]
    second: Optional[Candidate]


def _parse_candidate(obj: object) -> Optional[Candidate]:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise TypeError(f"candidate must be dict or null, got {type(obj)}")
    return Candidate(
        predictor=int(obj["predictor"]),
        filter=int(obj["filter"]),
        reorder=int(obj["reorder"]),
        interleave=int(obj["interleave"]),
        entropy=int(obj["entropy"]),
        bits=int(obj["bits"]),
    )


def iter_tlg8_training_jsonl(path: Path) -> Iterator[TrainingRow]:
    with path.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield TrainingRow(
                image=str(row["image"]),
                image_size=(int(row["image_size"][0]), int(row["image_size"][1])),
                tile_origin=(int(row["tile_origin"][0]), int(row["tile_origin"][1])),
                block_origin=(int(row["block_origin"][0]), int(row["block_origin"][1])),
                block_size=(int(row["block_size"][0]), int(row["block_size"][1])),
                components=int(row["components"]),
                pixels=[int(x) for x in row["pixels"]],
                best=_parse_candidate(row.get("best")),
                second=_parse_candidate(row.get("second")),
            )
```

---

## 2. `--label-cache-bin`（固定長バイナリ）

### 概要

- 先頭ヘッダ等はなく、`LabelRecord`（128 バイト）をブロックごとに **そのまま連結**したファイルです。
- 書き込みは `wb`（上書き）です。
- C++ 側は `#pragma pack(push, 1)` の構造体を `fwrite()` しています（`src/label_record.h`）。
- 実装上はリトルエンディアン環境前提になっています（Python 側も `<` で読むのが現実的です）。

### `LabelRecord` レイアウト（128 バイト）

構造体は以下（パディング含め固定 128 バイト）：

|オフセット|サイズ|型|名前|内容|
|---:|---:|---|---|---|
|0|4|`uint32`|`magic`|常に `0x4C424C38`（ASCII で `"LBL8"`）|
|4|2|`uint16`|`version`|現在は `1`|
|6|2|`uint16`|`reserved`|現在は `0`|
|8|24|`int16[12]`|`labels`|best/second の各パラメータ|
|32|4|`uint32`|`crc32`|現在は `0`（未使用）|
|36|92|`uint8[92]`|`padding`|現在は `0`（未使用）|

### `labels[12]` の意味

`labels` は `int16` 配列で、未設定は `-1` が入ります。

|index|意味|
|---:|---|
|0|`best_predictor`|
|1|`best_filter_perm`|
|2|`best_filter_primary`|
|3|`best_filter_secondary`|
|4|`best_reorder`|
|5|`best_interleave`|
|6|`second_predictor`（存在しない場合は -1）|
|7|`second_filter_perm`|
|8|`second_filter_primary`|
|9|`second_filter_secondary`|
|10|`second_reorder`|
|11|`second_interleave`|

`filter` は JSONL では 0–95 の「1つのコード」ですが、label-cache では以下の関数相当で 3 つに分解して格納します（実装 `split_filter()`）：

```text
perm      = ((code >> 4) & 0x7) % 6
primary   = ((code >> 2) & 0x3) % 4
secondary = (code & 0x3) % 4
```

### Python で読む例（struct で固定長を反復）

```python
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


_LABEL_RECORD = struct.Struct("<IHH12hI92s")  # 128 bytes
_MAGIC = 0x4C424C38  # "LBL8"


@dataclass(frozen=True)
class LabelEntry:
    predictor: int
    filter_perm: int
    filter_primary: int
    filter_secondary: int
    reorder: int
    interleave: int


@dataclass(frozen=True)
class LabelRecordPy:
    best: LabelEntry
    second: LabelEntry | None


def iter_label_cache_bin(path: Path) -> Iterator[LabelRecordPy]:
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(_LABEL_RECORD.size)
            if not chunk:
                return
            if len(chunk) != _LABEL_RECORD.size:
                raise ValueError("LabelRecord の途中でファイルが終わりました")

            magic, version, reserved, *rest = _LABEL_RECORD.unpack(chunk)
            labels = rest[:12]  # type: ignore[assignment]
            # crc32 = rest[12]  # 現状未使用
            # padding = rest[13]  # 現状未使用

            if magic != _MAGIC:
                raise ValueError(f"magic 不一致: 0x{magic:08x}")
            if version != 1:
                raise ValueError(f"version 不明: {version}")
            if reserved != 0:
                raise ValueError(f"reserved が 0 ではありません: {reserved}")

            best = LabelEntry(
                predictor=int(labels[0]),
                filter_perm=int(labels[1]),
                filter_primary=int(labels[2]),
                filter_secondary=int(labels[3]),
                reorder=int(labels[4]),
                interleave=int(labels[5]),
            )
            if int(labels[6]) < 0:
                second = None
            else:
                second = LabelEntry(
                    predictor=int(labels[6]),
                    filter_perm=int(labels[7]),
                    filter_primary=int(labels[8]),
                    filter_secondary=int(labels[9]),
                    reorder=int(labels[10]),
                    interleave=int(labels[11]),
                )
            yield LabelRecordPy(best=best, second=second)
```

---

## 3. `--label-cache-meta`（メタデータ JSON）

### 概要

`--label-cache-bin` とセットで生成される、整合性チェックと再現性のためのメタデータです。

主な役割：

- `record_size` / `record_count` の提示（バイナリの期待サイズを計算できる）
- 入力ファイル（データセット）の列挙と各ファイルの SHA-256
- 入力全体をまとめた `dataset_sha256`（順序付きハッシュ）

### JSON のスキーマ（schema=1）

トップレベルは以下のキーを持ちます：

- `schema`：`1`（整数）
- `record_size`：`128`（整数）
- `record_count`：レコード数（整数）
- `inputs`：入力ファイル配列（配列、空の場合あり）
  - `path`：正規化済みパス（文字列。`canonical` が取れればそれ、無理なら `absolute`）
  - `size`：ファイルサイズ（整数）
  - `mtime`：更新時刻（秒、`float`。UNIX epoch 基準の `time.time()` 相当）
  - `sha256`：ファイル内容の SHA-256（16進 64 文字）
- `dataset_sha256`：入力全体をまとめた SHA-256（16進 64 文字）

また、メタ生成時に **バイナリ実サイズが `record_count * 128` と一致することを検証**してから書き出します。

### `dataset_sha256` の計算方法

入力 `inputs` を先頭から順に処理し、次のバイト列を SHA-256 に投入していった最終ハッシュです：

1. 各入力の `sha256`（16進文字列）を 32 バイトにデコードしたもの
2. `size` を **8 バイト little-endian**で表現したもの
3. `path` の UTF-8 バイト列（区切り文字なし）

`inputs` が空の場合、`dataset_sha256` は SHA-256 の空入力ハッシュになります。

### Python で読む例（整合性チェック＋dataset_sha256 再計算）

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelCacheMetaInput:
    path: str
    size: int
    mtime: float
    sha256: str


@dataclass(frozen=True)
class LabelCacheMeta:
    schema: int
    record_size: int
    record_count: int
    inputs: list[LabelCacheMetaInput]
    dataset_sha256: str


def load_label_cache_meta(path: Path) -> LabelCacheMeta:
    obj = json.loads(path.read_text(encoding="utf-8"))
    inputs = [
        LabelCacheMetaInput(
            path=str(x["path"]),
            size=int(x["size"]),
            mtime=float(x["mtime"]),
            sha256=str(x["sha256"]),
        )
        for x in obj.get("inputs", [])
    ]
    return LabelCacheMeta(
        schema=int(obj["schema"]),
        record_size=int(obj["record_size"]),
        record_count=int(obj["record_count"]),
        inputs=inputs,
        dataset_sha256=str(obj["dataset_sha256"]),
    )


def compute_dataset_sha256(meta: LabelCacheMeta) -> str:
    h = hashlib.sha256()
    for inp in meta.inputs:
        h.update(bytes.fromhex(inp.sha256))
        h.update(int(inp.size).to_bytes(8, "little", signed=False))
        h.update(inp.path.encode("utf-8"))
    return h.hexdigest()


def validate_label_cache(bin_path: Path, meta_path: Path) -> None:
    meta = load_label_cache_meta(meta_path)
    if meta.schema != 1:
        raise ValueError(f"schema 不明: {meta.schema}")
    if meta.record_size != 128:
        raise ValueError(f"record_size が想定外: {meta.record_size}")

    expected = meta.record_size * meta.record_count
    actual = bin_path.stat().st_size
    if actual != expected:
        raise ValueError(f"bin サイズ不一致: actual={actual} expected={expected}")

    recomputed = compute_dataset_sha256(meta)
    if recomputed != meta.dataset_sha256:
        raise ValueError("dataset_sha256 が一致しません")
```

---

## 4. JSONL と label-cache を突き合わせる（同一実行の前提）

同一実行で `--tlg8-dump-training` と `--label-cache-*` を同時に出した場合は、ブロック順が一致するため、Python 側で次のように 1:1 に対応付けできます。

```python
from pathlib import Path


def iter_aligned(dump_jsonl: Path, label_bin: Path):
    from itertools import zip_longest

    for i, (row, rec) in enumerate(
        zip_longest(iter_tlg8_training_jsonl(dump_jsonl), iter_label_cache_bin(label_bin)),
        start=0,
    ):
        if row is None or rec is None:
            raise ValueError("行数とレコード数が一致しません")
        yield i, row, rec
```

注意：

- label-cache は **座標や画像タグを持たない**ため、単体で「どのブロックのラベルか」を復元するには、別途同一順序の情報源（JSONL など）が必要です。
- JSONL は追記モードなので、複数回実行の混在に注意してください（`meta.inputs` でダンプファイルが含まれる設定の場合、整合性チェックに利用できます）。
