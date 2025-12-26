# AGENTS.md — TLG8 ML Candidate-Reduction Pipeline (CODEX)

This document defines a **fully-automatable** machine-learning pipeline for TLG8.

## Goal

TLG8 ML is **NOT** optimizing compression ratio directly. The purpose is:

* Use ML to **narrow encoder parameter candidates to top-N**
* Reduce **total wall-clock encode time** by reducing exhaustive search

Primary metrics:

* **Accuracy:** block-micro hit-rate (see below)
* **Speed proxy:** minimize **#params** (model parameter count)

## Repository Layout (MUST)

* All scripts go under: `ml/`
* All run outputs, artifacts, and logs go under: `ml/runs/`

### Required output structure per run

Each run must create a unique directory:

* `ml/runs/<run_id>/`

  * `config.json` (all knobs used)
  * `progress.jsonl` (append-only; one line per trial)
  * `best.json` (current best summary)
  * `artifacts/`

    * trained model weights + exported inference bundles
    * feature pipeline metadata
    * evaluation reports
  * `splits/`

    * split manifest (fixed, reproducible)

### Progress logging and resume (MUST)

The agent must be robust against API stops or manual pauses.

* Write an **append-only** trial log at: `ml/runs/<run_id>/progress.jsonl`
* Each trial must record:

  * trial_id, timestamp
  * git commit hash
  * dataset identifiers and counts
  * feature set specification (selected raw features + derived features)
  * MLP architecture for each head
  * training hyperparameters (seed, max_epochs, patience, lr, etc.)
  * valid loss (for early stopping), valid hit-rate
  * total #params per head and total
  * any failure mode / exception summaries
* On start, the pipeline must:

  * locate an existing `ml/runs/<run_id>/progress.jsonl` if resuming
  * load the **best** results so far
  * continue from the **next** trial_id

## Data Sources

Input artifacts are produced by the C++ extractor and pack step.

* `training.all.jsonl` (concatenated; contains block features + best/second metadata)
* `labels.all.bin` (LabelRecord array; fixed-order)
* `index.jsonl` (image offsets)
* `labels.meta.json` (integrity metadata)

**Order guarantee:** `training.all.jsonl` and `labels.all.bin` are aligned in identical block order.

### LabelRecord format

`labels` is `int16[12]`, unset = `-1`.

| idx | meaning                 |
| --: | ----------------------- |
|   0 | best_predictor          |
|   1 | best_filter_perm        |
|   2 | best_filter_primary     |
|   3 | best_filter_secondary   |
|   4 | best_reorder            |
|   5 | best_interleave         |
|   6 | second_predictor        |
|   7 | second_filter_perm      |
|   8 | second_filter_primary   |
|   9 | second_filter_secondary |
|  10 | second_reorder          |
|  11 | second_interleave       |

Filter split (matches `split_filter()`):

* `perm      = ((code >> 4) & 0x7) % 6`
* `primary   = ((code >> 2) & 0x3) % 4`
* `secondary = (code & 0x3) % 4`

### Bits sidecar (MUST)

Create and use:

* `ml/runs/<run_id>/dataset/bits.all.npy`

Spec:

* dtype: `uint64`
* shape: `[N, 2]`
* col0: `bits_best`
* col1: `bits_second`
* Block order MUST match `labels.all.bin`.

## Dataset Filtering Rules (MUST)

A block is eligible iff:

* `block_size == [8, 8]`
* `best != null` and `second != null`
* `second` differs from `best` (the full (predictor, filter, reorder, interleave) must not be identical)
* `components` can be 3 or 4, but:

  * for `components==4`, drop alpha and use RGB only

Entropy is NOT part of ML search:

* `entropy` is fixed to **Plain (0)**

## Feature Policy

### Pixels

* Allowed: use raw `pixels` as-is.
* Flatten order is `y → x → comp`.

  * components==3: `RGBRGB...`
  * components==4: `ARGBARGB...` (alpha dropped; keep RGB)
* Normalize: `pixel_float = pixel_uint8 / 255.0`.

### Additional features

* Use non-pixel numeric features present in JSONL (e.g., `reorder_tv_mean`, `reorder_tv2_mean`, etc.).

### Feature search (agent allowed)

The agent is allowed to:

* select/drop raw feature columns
* apply **lightweight** derived transforms only:

  * unary: log1p, sqrt, square, abs, clip
  * limited pairwise: sum/diff/product/ratio
  * quantile binning

Hard constraint:

* total feature dimension after derivation must be **≤ 256** (excluding pixels; pixels are fixed 192)

## Model Architecture

### Multi-stage decision flow (fixed)

`predictor → cf_perm → cf_primary → cf_secondary → reorder → interleave`

Heads have class counts:

* predictor: 8
* cf_perm: 6
* cf_primary: 4
* cf_secondary: 4
* reorder: 8
* interleave: 2

### Beam widths (fixed)

* predictor: keep top **4**
* cf_perm: keep top **4**
* cf_primary: keep top **3**
* cf_secondary: keep top **2**
* reorder: keep top **4**
* interleave: keep top **2**

Final candidate count: `4 × (4×3×2) × 4 × 2 = 768` full tuples.

### Models (fixed)

* Each head is an independent **MLP**.
* Each head outputs logits for all classes in one forward pass.
* Tie-break: deterministic by class id ascending.

### Optimization objective

* Primary: meet accuracy constraint
* Secondary: minimize total `#params` across all heads

## Training Targets (Projection)

### Scalar minimized for projection

Define:

* `score := bits_best` (from `bits.all.npy[:,0]`)

`margin := bits_second - bits_best` is optional for analysis only.

### Best-case projection (fixed)

For each head and each class value `v`, define projected score:

* `proj_score_head[v] = min(score | head_value == v)`

This is computed independently per head (no conditional projection).

### Head ground-truth set per sample

For each head, the positive set is:

* the **top-2** class values with smallest `proj_score_head[v]` (rank-based; no threshold)

### Loss (fixed)

Soft-target cross entropy:

* Assign 0.5 / 0.5 to the two positive classes; 0 to others.

## Train/Validation Split (fixed)

* Split unit: **block**
* Random split with fixed seed:

  * train 90%
  * valid 10%
* Split must be saved to `ml/runs/<run_id>/splits/split.json` for reproducibility.

## Early Stopping and Model Selection

* Early stopping monitors: **valid CE loss**
* Best model selection uses: **valid hit-rate**

## Inference Scoring (fixed)

### Combine head logits

For a full tuple, combine by simple sum:

`S(full) = S_pred[p] + S_perm[perm] + S_primary[primary] + S_secondary[secondary] + S_reorder[r] + S_interleave[i]`

Higher is better.

### Output

Return top-3 full tuples (predictor, filter_code 0–95, reorder, interleave).

Filter code is reconstructed from (perm, primary, secondary) using the inverse mapping consistent with `split_filter()` (implementation detail).

Entropy is fixed to Plain.

## Evaluation

### Block-micro hit-rate (primary metric)

A prediction is a hit if:

* among the returned **top-3 full tuples**, at least one matches either of the block’s **true top-2** full tuples.

### Hard constraint

* Valid hit-rate must be **≥ 90%**.

## Trial Definition (fixed)

One **trial** means:

* choose a feature pipeline (≤256 derived dims + pixels)
* choose MLP architectures for all heads (layers/widths)
* train all heads with early stopping
* evaluate hit-rate and #params

## Global Stopping Condition (failure)

If the best valid hit-rate fails to improve by **≥ 0.1%** for **10** consecutive trials:

* stop the search and mark the plan as failed.

## Implementation Tasks (for CODEX)

1. `ml/make_bits_sidecar.py`

   * Input: `training.all.jsonl`
   * Output: `bits.all.npy` in run dataset dir

2. `ml/dataset_builder.py`

   * Input: `training.all.jsonl`, `labels.all.bin`, `bits.all.npy`
   * Output: tensors/arrays suitable for training
   * Enforce filtering rules

3. `ml/train_rankers.py`

   * Implements trial loop, feature search, MLP architecture search (bounded)
   * Writes progress logs and supports resume

4. `ml/infer_rankers.py`

   * Loads trained head models
   * Applies beam search
   * Returns top-3 full tuples

5. `ml/eval_rankers.py`

   * Computes block-micro hit-rate

## Notes

* Keep every run fully reproducible (fixed seed and saved split).
* Never overwrite artifacts; create new run_id if configuration changes materially.
* Always write append-only logs for recoverability.


* 元画像から 生成した label-cache と training-json は ml/runs/ の下にあります。
* 機械学習タスクでは CUDA を使用するようにしてください。