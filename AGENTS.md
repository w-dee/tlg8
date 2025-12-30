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

In this repository, the packed artifacts are located at:

* `ml/source/packed_out/`

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

Note: `bits_best/bits_second` are present in `training.all.jsonl` under `best.bits` / `second.bits`.

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

### Neighbor pixels (policy)

* Do **NOT** add any additional neighbor-pixel context beyond what is already present in the dataset.
  * In particular, do not expand to larger neighborhoods (e.g., more rows/cols, multi-block context, or wider padding windows).
  * If more signal is needed, prefer adding **candidate-specific cheap score dumps** (see below) rather than adding more neighbor pixels.

### Additional features

* Use non-pixel numeric features present in JSONL (e.g., `reorder_tv_mean`, `reorder_tv2_mean`, etc.).
* Prefer adding per-block **candidate-specific cheap score** features computed by the encoder (e.g., residual energy per predictor, min filtered energy per filter code).

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

Return top-32 full tuples (predictor, filter_code 0–95, reorder, interleave).

Filter code is reconstructed from (perm, primary, secondary) using the inverse mapping consistent with `split_filter()` (implementation detail).

Entropy is fixed to Plain.

## Evaluation

### Block-micro hit-rate (primary metric)

A prediction is a hit if:

* among the returned **top-32 full tuples**, at least one matches either of the block’s **true top-2** full tuples.

### Hard constraint

* Valid hit-rate must be **≥ 95%** (hit@32).

## Relaxation Proposals (NON-DEFAULT)

This section is a concrete proposal for alternative objectives/modes beyond the default spec above.

**Compatibility rule:** the strict spec above remains the default. A relaxed mode must be explicitly enabled (e.g., by adding a `--mode` flag to training/eval scripts and saving it into `bundle.json`/`config.json`). Existing runs and bundles remain readable; new modes must not break strict-mode behavior.

### Mode A — Candidate-Reduction / Operational (Relax `top-3`)

**What to relax**
* **Output:** allow returning top-`K` tuples instead of fixed top-3. Suggested `K ∈ {8,16,32}`.
* **Primary metric:** use hit@K instead of hit@3; keep hit@3 as a secondary metric for regressions.
* **Hard constraint:** replace “valid hit-rate ≥ 90% (hit@3)” with a mode-specific threshold, e.g.:
  * `hit@16 ≥ 90%` (or `hit@32 ≥ 95%`) as the primary gate, plus
  * `#candidates` (K) bounded as the speed proxy.

**Why this helps**
* Empirically, increasing beam sizes often improves hit@10/hit@K without moving hit@3, suggesting the model can retrieve good candidates but not reliably place them in the top-3.

**Risks**
* Higher K reduces speed gains and may increase downstream encode time variance.
* If encoder search cost is non-linear in candidate count, K increases may be expensive.

**Verification steps**
* Add `--topk` (or `--eval-topn`) to `ml/infer_rankers.py` and `ml/eval_rankers.py` and log `hit@{3,8,16,32}` side-by-side.
* Validate on the same split: ensure `hit@K` monotonically increases with K and no unexpected regressions occur at hit@3.
* End-to-end proxy: optionally measure encoder wall-clock on a fixed image set for K values (8/16/32).

**Current recommended bundle (distilled student)**
* Use `ml/runs/codex_filteri_student_64_32_20251230a/artifacts/trial_0000/bundle.json` (hidden sizes `[64,32]`).
  * Source: distilled from `ml/runs/codex_distill_filteri_20251230a/artifacts/trial_0001/bundle.json`.
  * Reported valid: `hit@32 ≈ 98.125%`, `hit@16 ≈ 96.655%`.
* To promote a different bundle into a new run dir: `python ml/promote_bundle.py --src-bundle <path/to/bundle.json> --run-id <new_run_id>`.

### Mode B — Two-Stage Re-Rank (Relax “independent heads only”)

**What to relax**
* Keep a light Stage-1 model to generate top-`M` candidates (e.g., 512–4096), then use a heavier Stage-2 model to re-rank and output top-3 (or top-K).
* Stage-2 can use richer features (including candidate-dependent scores) while keeping Stage-1 lightweight.

**Compatibility**
* Stage-1 remains compatible with current `bundle.json` schema; Stage-2 introduces an optional `reranker` section in the bundle.
* Inference stays deterministic: Stage-2 must preserve deterministic tie-break by tuple id.

**Risks**
* Implementation complexity and higher inference cost (but still far below exhaustive search if M is bounded).
* More moving parts: failure modes in candidate enumeration or feature alignment.

**Verification steps**
* Add `ml/train_reranker.py`, `ml/infer_reranker.py`, `ml/eval_reranker.py` (or integrate into existing scripts behind `--mode rerank`).
* Unit checks: reranker input candidates are stable across runs; no tuple reconstruction mismatch.
* Compare: strict head-only top-3 vs rerank top-3 on the same valid split; track both hit@3 and “bits regret” (see Mode C).

### Mode C — Objective/Labeling Relaxation (Optimize ranking, not projection top-2)

**What to relax**
* Replace the per-head projection target with a training objective aligned to the final tuple ranking:
  * sampled listwise/pairwise ranking (e.g., sampled softmax / contrastive) over candidate tuples, or
  * joint scorer over `(predictor, filter_i, reorder)` (or full tuple) with candidate sampling.
* Add a secondary metric that matches candidate-reduction utility:
  * **bits regret@K:** `min(bits among predicted top-K) - bits_best` (mean/median/95p).

**Compatibility**
* Keep `bits.all.npy` alignment invariant; log any new sampling strategy and candidate universe definition into `progress.jsonl`.

**Risks**
* Sampling bias: poor negative sampling can inflate metrics without real gains.
* More GPU memory pressure; higher risk of training instability.

**Verification steps**
* Track: hit@{3,8,16}, bits regret@{3,8,16}, and candidate count.
* Sanity: bits regret@K should be ≤ 0 when the true best is retrieved; distribution should improve with K.
* Reproducibility: fixed seed sampling and a saved candidate sampling manifest (optional) for exact reruns.

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

## Verification (MUST RUN)

Codex MUST run verification commands after making changes and before declaring completion.

Required:
- python -m compileall ml
- python ml/train_rankers.py --run-id <run_id> --smoke


## Notes

* Keep every run fully reproducible (fixed seed and saved split).
* Never overwrite artifacts; create new run_id if configuration changes materially.
* Always write append-only logs for recoverability.


* 学習元画像から 生成した label-cache と training-json は ml/runs/ の下にあります。
* 機械学習タスクでは CUDA を使用するようにしてください。

Local defaults / constraints for this workspace:

* Input packed dataset: `ml/source/packed_out/`
* Budget: up to 60 minutes per trial, up to 100 trials total
* GPU: default CUDA device (RTX 3070 Ti 16GB)
