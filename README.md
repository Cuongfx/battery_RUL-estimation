# Battery RUL Classification — BatteryML pipeline

Sliding-window RUL (Remaining Useful Life) classification of lithium-ion cells from the [BatteryML](https://github.com/microsoft/BatteryML) corpus. A window of consecutive cycles is classified into one of 5 RUL bands.

> **Fork note** — this is a bug-fix and slim-down of [PhanLeSon03/battery_estimation](https://github.com/PhanLeSon03/battery_estimation). The upstream MIT-Stanford pipeline has been removed and the BatteryML path rewritten; see [document/what_changed.md](document/what_changed.md) for the measured differences.

## RUL class boundaries

| Class | Condition | Meaning |
|-------|-----------|---------|
| 0 | RUL > 400 | Early life |
| 1 | 300 < RUL ≤ 400 | Mid-early life |
| 2 | 200 < RUL ≤ 300 | Mid life |
| 3 | 100 < RUL ≤ 200 | Late life |
| 4 | RUL ≤ 100 | Near end of life |

Boundaries live in one place (`RUL_EDGES` in `dataset_clf_bml_v2.py`); class count and report labels are derived from them.

---

## Table of Contents
- [Pipeline](#pipeline)
- [Usage](#usage)
- [Automation scripts](#automation-scripts)
  - [On a SLURM cluster](#on-a-slurm-cluster)
  - [Cluster setup](#cluster-setup)
- [Model architectures](#model-architectures)
- [Feature engineering](#feature-engineering)
- [File structure](#file-structure)
- [Known issue](#known-issue)
- [Citation](#citation)

---

## Pipeline

```
BatteryML .pkl  ──gen_feature_bml_v2──►  .npz  ──dataset_clf_bml_v2──►  DataLoader
                                                                            │
                        ┌───────────────────────────────────────────────────┤
                        ▼                        ▼                          ▼
              train_clf_bml_V2      train_clf_bml_transformer_V2   train_clf_es_bml_V2
                (CNN + BiGRU)          (CNN + Transformer)        (sparse CMA-ES on GRU)
                        │                        │                          │
                        └────────────────────────┴──────────────────────────┘
                                                 ▼
                                    predict_clf_bml_V2.ipynb
```

---

## Usage

### 1 — Extract features

```bash
python gen_feature_bml_v2.py --data_dir ./Raw/Raw_BML/MATR --out_dir ./content_bml/MATR
```

One `.npz` per cell, mirroring the source directory layout. Runs once.

End-of-life is the first cycle below 80 % of initial capacity, confirmed by a window of following cycles — a single-cycle dip no longer ends a cell's life. Pass `--eol_confirm 0` for the old first-crossing rule.

Cells whose logging stopped well before EOL are kept, not dropped: their labels are compressed by a median of 17 cycles (max 79, under one class width), while dropping them would cost 72 % of the corpus. Each `.npz` carries `eol_reached` and `retention` so the decision can be revisited.

### 2 — Inspect the dataset (optional)

```bash
python dataset_clf_bml_v2.py --content_dir ./content_bml/MATR
```

Builds the three dataloaders and prints per-split sample counts, class balance and tensor shapes. Writes nothing.

### 3 — Train

```bash
# CNN + BiGRU
python train_clf_bml_V2.py --content_dir ./content_bml/MATR \
    --output_dir ./checkpoints_clf_bml_MATR

# CNN + Transformer
python train_clf_bml_transformer_V2.py --content_dir ./content_bml/MATR \
    --output_dir ./checkpoints_clf_bml_MATR_tf

# Sparse CMA-ES on the GRU weights, starting from a trained checkpoint
python train_clf_es_bml_V2.py --content_dir ./content_bml/MATR \
    --output_dir ./checkpoints_clf_bml_MATR_es \
    --pretrain_ckpt ./checkpoints_clf_bml_MATR/best_clf_bml.pt
```

Each writes `best_clf_bml.pt`, `model_config.json`, both scalers, test predictions and `Info_log.txt` into `--output_dir`.

`train_clf_bml_V2.py` also accepts `--es_mode`: `0` gradient only (default), `1` CMA-ES afterwards, `2` on stagnation, `3` CMA-ES alone. Modes 1–3 need `pip install cma`.

### 4 — Inference and plots

Open `predict_clf_bml_V2.ipynb`, set `dataset` and `CKPT_DIR`, run all. Produces the per-cell sliding-window plot, an all-cells grid, a cycle-life overlay, NEOL parity and error histogram, a per-cell table and confusion matrices.

---

## Automation scripts

PowerShell wrappers that chain the CLI calls above for batch runs, kept in `RunToTrain/`. Each accepts `-ExecutionPolicy Bypass -File .\RunToTrain\<script>.ps1`, `cd`s to the repo root on its own (so it works no matter where you launch it from) and prints its own progress/summary; a non-zero Windows teardown exit code from CUDA (see [Known issue](#known-issue)) is treated as benign, not a failure.

| Script | Purpose | Trains |
|---|---|---|
| `RunToTrain/run_dataset.ps1` | Scans every raw `.pkl` folder under `Raw/Raw_BML/` and runs `gen_feature_bml_v2.py` on each, producing the `.npz` dataset in `content_bml/<folder>`. Run this first — nothing else works without it. | — (feature export only) |
| `RunToTrain/run_train_all_default.ps1` | Trains the default model (`train_clf_bml_V2.py`, CNN+BiGRU) on each folder's already-built dataset, with default hyperparameters. | 1 model x N folders |
| `RunToTrain/run_train_all_model.ps1` | Trains all three architectures — default (`train_clf_bml_V2.py`), CMA-ES (`train_clf_es_bml_V2.py`), and Transformer (`train_clf_bml_transformer_V2.py`) — selected via `-TrainScript 1/2/3` (or an interactive prompt if omitted). CMA-ES needs a checkpoint already produced by `-TrainScript 1`. | 1 of 3 models x N folders per run |
| `RunToTrain/run_model_adjust.ps1` | Grid search over model width/depth — `cnn_dim` x `gru_dim` x `gru_layers` — training `train_clf_bml_V2.py` repeatedly on one dataset (`-ContentDir`, default `./content_bml/LFP`) to see how CNN/GRU layer sizing affects accuracy. | 1 model, many configs |
| `RunToTrain/run_exp_15.ps1` | Grid search over the input window — `N_EARLY` x `N_RANDOM` (values 2–10) — on one dataset (`-ContentDir`, default `./content_bml/MATR`), 5 repeats per combo, to see how much lead-in vs. random-window history the model needs. | 1 model, many configs |

Typical order: `RunToTrain\run_dataset.ps1` → `RunToTrain\run_train_all_default.ps1` (or `RunToTrain\run_train_all_model.ps1` for all three architectures) → `RunToTrain\run_model_adjust.ps1` / `RunToTrain\run_exp_15.ps1` for hyperparameter sweeps once a baseline works.

### On a SLURM cluster

`RunToTrainServer/` mirrors the five scripts above for a SLURM batch system. Same names, same grids, same output layout — the difference is that nothing runs on the login node: each script writes a job directory under `jobs/<name>_<timestamp>/` holding a `params.txt` (one python command per line) and a `job.sh`, then submits them as a single **array job**, so the 144-run and 125-run grids queue as 144 and 125 tasks instead of one 10-hour serial job.

```bash
pwsh ./RunToTrainServer/run_dataset.ps1                       # 3 tasks, CPU only
pwsh ./RunToTrainServer/run_train_all_default.ps1             # 3 tasks, 1 GPU each
pwsh ./RunToTrainServer/run_train_all_model.ps1 -TrainScript 2
pwsh ./RunToTrainServer/run_exp_15.ps1 -Throttle 8            # 125 tasks, 8 at a time
```

Needs `pwsh` (PowerShell 7) on the cluster — see [Cluster setup](#cluster-setup). Every script takes `-Account`, `-Qos`, `-Gpu`, `-Cpus`, `-Mem`, `-Time`, `-CondaEnv` and `-Throttle` (how many array tasks may run at once), plus `-DryRun` to write the job script and print the `sbatch` line without submitting. The `module load` and conda bootstrap lines live in `RunToTrainServer/_SlurmCommon.ps1` — change them there once, not in five files.

`jobs/` and `logs/` are generated and not committed.

### Cluster setup

`pwsh` is not installed on most HPC systems. Check, then install it into your conda environment if it is missing:

```bash
which pwsh || module avail 2>&1 | grep -i powershell
conda activate battery_ml
conda install -c conda-forge powershell_core     # provides `pwsh`
```

If neither works, unpack the official tarball into your home directory and add it to `PATH`. Beyond that, the cluster needs: the repo cloned, `Raw/Raw_BML/` and `content_bml/` copied over (`rsync -avz`), and a conda environment matching `-CondaEnv`.

---

## Model architectures

Input per sample: **16 cycles** = first `N_EARLY = 8` (fresh-cell baseline) + `N_RANDOM = 8` consecutive from anywhere past them.

### CNN + BiGRU — `model_clf.py`

```
dq (B, 16, 1, 1000) ──► time-distributed Conv1d ──► (B, 16, cnn_dim)
                          3 × [Conv1d → BN → ELU → Dropout → MaxPool]
                          AdaptiveAvgPool1d(1)

summary (B, 16, 12) ──► Linear → ELU → Dropout ──► (B, 16, cnn_dim)

concat ──► (B, 16, cnn_dim×2) ──► Dropout
       ──► BiGRU × 2 ──► concat(h_n[-2], h_n[-1]) ──► (B, gru_dim×2)
       ──► Linear → ELU → Dropout → Linear ──► (B, 5)
```

### CNN + Transformer — `train_clf_bml_transformer_V2.py`

Same CNN front end (GELU instead of ELU), then:

```
concat ──► Linear ──► (B, 16, d_model)
       ──► + sinusoidal positional encoding
       ──► TransformerEncoder (Pre-LN) × n_enc_layers
       ──► mean pool ──► Linear → GELU → Dropout → Linear ──► (B, 5)
```

The position table is a non-persistent buffer: derived from `(max_seq_len, d_model)`, recomputed on load, absent from `state_dict`.

### Losses

`OrdinalLoss` (default) treats the classes as ordered, minimising binary cross-entropy over cumulative thresholds `P(y > k)`. `--loss cross_entropy` selects plain cross-entropy.

> Both decode predictions with `argmax`. `ordinal_predict`, the matching cumulative decoder, disagrees with `argmax` on 37 % of samples and is currently unused — switching decoders would change every published number, so it is left as a deliberate open question rather than a silent fix.

---

## Feature engineering

### Summary vector — 12 features per cycle

| # | Feature | Description |
|---|---------|-------------|
| 1 | `Qd` | Discharge capacity (Ah) |
| 2 | `c_t` | Charge duration (s) |
| 3 | `dc_t` | Discharge duration (s) |
| 4 | `dqdv_slope_max` | Max slope of the dQ/dV curve |
| 5 | `dqdv_slope_min` | Min slope of the dQ/dV curve |
| 6 | `dqdv_min` | Min of the dQ/dV curve |
| 7 | `dqdv_avg` | Mean of the dQ/dV curve |
| 8 | `20·log₁₀(std(Qd))` | Log-std of discharge capacity within cycle |
| 9 | `20·log₁₀(std(Qc))` | Log-std of charge capacity within cycle |
| 10 | `20·log₁₀(std(Id))` | Log-std of discharge current |
| 11 | `20·log₁₀(std(Ic))` | Log-std of charge current |
| 12 | position | Real cycle number ÷ `POS_SCALE` |

`Qc` is exported and loaded but excluded from the model vector: adding it widens the input to 13 and invalidates existing checkpoints. Enable it in one line (`SUMMARY_KEYS`) when wanted.

The position feature uses the **logged cycle number**, not the array index. The two coincide on MATR but diverge wherever logging skips cycles — `HUST_7-5` starts at cycle 3, and the old code treated it as a fresh cell.

### dQ sequence — 1000 features per cycle

```
dQ[c] = qdlin[c] − qdlin[REF_CYCLE = 9]
```

`qdlin` is the discharge capacity resampled onto a fixed 1000-point voltage grid — an axis change, not a compression: total capacity is preserved exactly, while horizontal resolution inside the LFP plateau is not.

### Balanced sampling

Every cell contributes the same number of windows to each class, that count being the size of its own rarest class. A cell missing any class contributes nothing, so a cell must survive roughly 420 cycles for class 0 (`RUL > 400`) to exist for it. Cells dropped this way are now logged with the reason.

---

## File structure

```
battery_estimation/
├── Raw/Raw_BML/                    # BatteryML .pkl files (not committed)
├── content_bml/                    # extracted .npz features (generated)
├── checkpoints_clf_bml_*/          # per-run artefacts (generated)
│   ├── best_clf_bml.pt
│   ├── model_config.json
│   ├── dq_scaler_bml.pkl
│   ├── summary_scaler_bml.pkl
│   ├── clf_pred_bml.npy
│   ├── clf_true_bml.npy
│   └── Info_log.txt
├── gen_feature_bml_v2.py           # .pkl → .npz feature export
├── dataset_clf_bml_v2.py           # .npz → windowed, labelled dataloaders
├── model_clf.py                    # CNN+GRU model, OrdinalLoss, train/eval loops
├── train_utils.py                  # seeding, console+file logging, class names, model summary
├── train_clf_bml_V2.py             # gradient trainer (+ optional CMA-ES)
├── train_clf_bml_transformer_V2.py # transformer trainer, self-contained
├── train_clf_es_bml_V2.py          # sparse CMA-ES trainer
├── predict_clf_bml_V2.ipynb        # inference & visualisation
├── RunToTrain/                     # PowerShell batch-run wrappers, local Windows
├── RunToTrainServer/               # same wrappers, submitted as SLURM array jobs
├── jobs/                           # generated job.sh + params.txt per submission
├── logs/                           # SLURM stdout/stderr per array task (generated)
├── document/                       # what_changed.md and other reference docs
└── README.md
```

Both `gen_feature_bml_v2.py` and `dataset_clf_bml_v2.py` are organised in layers, each depending only on those below it — constants, labels, I/O, indexing, featurisation, dataset.

### What each file is responsible for

| File | Responsibility |
|---|---|
| `gen_feature_bml_v2.py` | Reads raw BatteryML `.pkl` cells, computes per-cycle scalars and the resampled `qdlin` curve, decides end-of-life, writes one `.npz` per cell. Runs once per dataset. |
| `dataset_clf_bml_v2.py` | Reads `.npz` files, slices them into fixed-length cycle windows, labels each window with a RUL class, builds train/val/test `DataLoader`s. Every trainer calls `build_clf_dataloaders` from here — none of them touch `.npz` files directly. |
| `model_clf.py` | The CNN+GRU architecture (`BatteryRULClassifier`), `OrdinalLoss`, and the generic `train_epoch` / `evaluate` loops. Used by both the GRU trainer and the CMA-ES trainer; the Transformer trainer defines its own model instead (different architecture, would gain nothing from sharing). |
| `train_utils.py` | The four pieces identical across every trainer: seeded RNG setup (`set_seed`), console+file log mirroring (`Tee`), RUL class names (`class_names`), and the layer-by-layer model summary printer (`print_model_summary`). Architecture-agnostic — none of the four know what model they're describing. |
| `train_clf_bml_V2.py` | Trains the CNN+GRU model by gradient descent, with CMA-ES available as an optional add-on (`--es_mode`) over the *whole* parameter space. |
| `train_clf_bml_transformer_V2.py` | Trains a CNN+Transformer model, self-contained (own model, own loss, own loops) so it pulls in nothing beyond the dataset layer. |
| `train_clf_es_bml_V2.py` | Trains the CNN+GRU model by sparse CMA-ES alone — no gradient descent. Only a fraction of the GRU's largest-magnitude weights evolve; everything else (CNN, summary projection, head) stays frozen at whatever a `--pretrain_ckpt` gave it, or at random init. |
| `predict_clf_bml_V2.ipynb` | Loads a checkpoint via its `model_config.json`, runs sliding-window inference across whole cell lifetimes, plots predictions, confusion matrices, and near-end-of-life estimates. |

### How the files depend on each other

```
gen_feature_bml_v2.py   model_clf.py   train_utils.py     (no local imports — leaves)
        │
        ▼
dataset_clf_bml_v2.py
        │
        ├──────────────┬──────────────────────────┐
        ▼              ▼                          ▼
train_clf_bml_V2.py   train_clf_es_bml_V2.py   train_clf_bml_transformer_V2.py
   (+ model_clf,          (+ model_clf,              (+ train_utils only —
      train_utils)           train_utils)               model defined inline)
        │              │                          │
        └──────────────┴──────────────────────────┘
                        ▼
              predict_clf_bml_V2.ipynb
                (+ model_clf, dataset_clf_bml_v2)
```

The arrows only point one way: `gen_feature_bml_v2` never imports from `dataset_clf_bml_v2`, and no trainer imports from another trainer. A change to a trainer can never leak into feature export or into a different trainer by accident.

### Which file to edit to change what

| To change | Edit | Propagates to | Requires |
|---|---|---|---|
| RUL class boundaries | `RUL_EDGES` in `dataset_clf_bml_v2.py` | `N_CLASSES` (derived), every trainer's model output size, `class_names()` in reports/notebook | Re-train — output layer width changes |
| Window length | `N_EARLY`, `N_RANDOM` in `dataset_clf_bml_v2.py` | `N_INPUT` (derived), every sample built by every trainer and the notebook | Re-train — different data, though model shapes are unaffected (GRU and Transformer both handle variable sequence length; the Transformer additionally needs `max_seq_len >= N_INPUT`) |
| Which per-cycle scalars feed the model | `SUMMARY_KEYS` in `dataset_clf_bml_v2.py` (subset of `SCALAR_KEYS` from the exporter) | `N_SUMMARY` (derived), the scaler's column count, every model's `summary_feats` input | Re-train — scaler and model input width both change |
| End-of-life rule | `EOL_FRACTION`, `EOL_CONFIRM`, `MAX_RETENTION` in `gen_feature_bml_v2.py` | `cycle_life` and every downstream label | Re-run `gen_feature_bml_v2.py` (new `.npz`), then re-train |
| Discharge-curve resolution | `V_BINS` in `gen_feature_bml_v2.py` | `qdlin`/`dq` array width | Re-run feature export. Model shapes are *not* affected — the CNN's `AdaptiveAvgPool1d(1)` collapses the curve to a fixed-size vector regardless of length — so old checkpoints still load, only the numbers going in differ |
| Reference cycle for the dQ curve | `REF_CYCLE` in `dataset_clf_bml_v2.py` | Every `dq` sample | Re-train |
| Model width/depth (`cnn_dim`, `gru_dim`, `d_model`, ...) | CLI flags on the relevant trainer | That trainer's `model_config.json` | Re-train that architecture only — the other trainers are untouched |
| Logging, seeding, class names, model summary format | `train_utils.py` | All three trainers simultaneously | Nothing to re-run by itself; re-train only if the change alters a seeded value |

`N_CLASSES`, `N_INPUT` and `N_SUMMARY` are never edited directly — they are computed from the constants above, so they cannot drift out of sync with them.

---

## Known issue

On Windows with CUDA, `train_clf_bml_V2.py` returns a non-zero exit code (`0xC0000409`) even on success. Traced with `faulthandler`: Python reaches the end, `atexit` handlers run, every artefact is written correctly — the fault occurs afterwards, while the CUDA libraries unload. Forcing CPU exits cleanly. This predates the rewrite and only matters when chaining commands or running CI.

---

## Citation

```bibtex
@inproceedings{10.1145/3711896.3737372,
author = {Tan, Ruifeng and Hong, Weixiang and Tang, Jiayue and Lu, Xibin and Ma, Ruijun and Zheng, Xiang and Li, Jia and Huang, Jiaqiang and Zhang, Tong-Yi},
title = {BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737372},
doi = {10.1145/3711896.3737372},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {5789–5800},
numpages = {12},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
