## What changed

Every claim below was verified by running the code, not by reading it.

### Dependency graph

Before, importing any BatteryML trainer pulled in the entire MIT pipeline:

```
train_clf_bml -> train_clf -> dataset_clf -> gen_features -> h5py
```

`train_clf.py` mixed two jobs in one file: the shared model definitions (lines 29–170) and its own MIT-dataset CLI. Python cannot import half a file, so borrowing the model meant loading the dataset layer of a corpus you were not using. The shared half is now `model_clf.py`, which imports nothing but `torch` and `numpy`.

```
train_clf_bml_V2      -> model_clf, dataset_clf_bml_v2   (no MIT modules)
train_clf_es_bml_V2   -> model_clf, dataset_clf_bml_v2
train_clf_bml_tf_V2   -> dataset_clf_bml_v2              (model inlined)
dataset_clf_bml_v2    -> gen_feature_bml_v2
model_clf             -> (nothing)
```

Repository went from 20 files to 7.

### Bugs fixed

**Feature export / dataset**

| Bug | Effect |
|---|---|
| `"Qc": qd` — charge capacity keyed to the discharge array | Silent duplicate of `Qd` the moment `Qc` is enabled |
| Loader re-ran its own EOL search over already-truncated arrays | Undid the exporter's confirmation-window rule on 4/169 MATR cells, one by up to 113 cycles |
| Missing `.npz` key returned an empty array | Feature column became a constant zero; training ran, accuracy silently dropped |
| `dqdv` loaded but never read | 194 MB of resident memory across 64 cells |
| Scaler fitted on one concatenated block | 2.56 GB peak; `partial_fit` in chunks peaks at 32 MB with bit-identical `transform` output |

**Sparse CMA-ES trainer**

| Bug | Effect |
|---|---|
| `mask = abs(w) <= threshold` under a comment reading *keep LARGE weights* | Evolved the **smallest** weights — the exact opposite of the design |
| `_ES_MODULES` listed all five submodules | 34/34 parameter tensors were maskable, so nothing was frozen despite the docstring |
| `train_loader=val_loader, val_loader=train_loader` at the call site | Fitness was measured on the training split |
| `cnn_dim` reconstructed as `head[0].in_features // 2` | Equals `gru_dim`, not `cnn_dim`; every worker crashed once the two differed |
| `except Exception: put((rank, 0.0))` with no logging | Worker crashes were indistinguishable from genuinely bad offspring |
| `best_acc` assigned once before the loop, then returned | Early-stop gate and return value both frozen at the pre-search value |

**Transformer trainer**

| Bug | Effect |
|---|---|
| `_build_pos_enc` defined, never called in `forward` | `f(x)` and `f(reversed x)` matched to **0.000000** — the window was an unordered bag |
| `classification_report` without `labels=` | `ValueError` whenever the test split lacked a class |
| No seeding anywhere | Two runs of the same command produced different models |

**Shared across trainers**

- `ReduceLROnPlateau(verbose=True)` — removed from PyTorch; the trainer crashed before the first epoch
- `import cma` at module level — blocked `--es_mode 0`, which uses no CMA-ES
- `torch.manual_seed` without `cudnn.deterministic` — `VaLoss` still drifted (0.8858 / 0.8850 / 0.8846); pinning cuDNN makes runs match digit for digit
- `va_acc >= best_val_acc` — re-saved the checkpoint on ties, so stagnation never accumulated
- CMA-ES branch left the model holding parameters that scored *worse* than the checkpoint on disk

### Added

- `model_config.json` beside every checkpoint, so the notebook rebuilds the architecture instead of retyping it
- `Info_log.txt` mirroring console output (verified byte-identical)
- Deterministic seeding across `random`, `numpy`, `torch`, CUDA and cuDNN
- Positional encoding actually applied in the transformer
