"""BatteryML RUL classification dataset (v2).

Consumes the .npz files written by ``gen_feature_bml_v2.py`` and yields
fixed-length cycle windows labelled by remaining useful life.

One sample is ``N_EARLY`` leading cycles -- a fresh-cell baseline every
sample carries -- followed by ``N_RANDOM`` consecutive cycles taken from
anywhere past them, ``N_INPUT`` cycles in total.

Layers, each depending only on the ones below it:

    L0 contract   constants shared with the exporter and the trainers
    L1 label      rul_to_class, window_rul
    L2 io         load_cell_npz, load_all_npz
    L3 index      build_sample_index
    L4 featurise  summary_row, CellCache, build_sample_tensors
    L5 dataset    BMLBatteryClsDataset, build_clf_dataloaders

How to run:
    python dataset_clf_bml_v2.py --content_dir ./content_bml

Build loaders for one BML family/subfolder only:
    python dataset_clf_bml_v2.py --content_dir ./content_bml/MATR

This module has no persisted output -- unlike the trainers, running it
writes nothing to disk. It is a smoke test: it builds the three dataloaders
and prints, per split:
    sample count and per-class breakdown (from build_sample_index)
    one batch's tensor shapes -- dq, summary, label
    scaler feature counts (dq_scaler, summary_scaler)
Import build_clf_dataloaders from here to get loaders inside a training
script instead of from the command line.
"""

from gen_feature_bml_v2 import NPZ_ALIAS, SCALAR_KEYS, V_BINS
import numpy as np
from pathlib import Path
import torch
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---- L0: contract ------------------------------------------------------

N_EARLY = 8     # Leading cycles carried by every sample.
N_RANDOM = 8    # Consecutive cycles drawn from past the lead-in.
N_INPUT = N_EARLY + N_RANDOM

RUL_EDGES = (400, 300, 200, 100)
N_CLASSES = len(RUL_EDGES) + 1

REF_CYCLE = 9

# Divisor turning a cycle number into the trailing position feature. Under
# the StandardScaler that follows, every positive divisor yields bit-wise
# identical output (measured across 5000, 2500, 2237 and 1.0: max |diff| = 0),
# so this is not a normalisation choice. It stays at 5000 purely so scalers
# stored in existing checkpoints remain usable.
POS_SCALE = 5000

# Summary scalars fed to the model, in fixed order: the scaler inside every
# checkpoint was fitted on exactly these columns in exactly this order, so
# permuting two entries breaks inference silently. "Qc" is loaded by L2 but
# deliberately excluded -- admitting it widens the vector to 13 and
# invalidates those checkpoints.
SUMMARY_KEYS = (
    "Qd", "c_t", "dc_t",
    "dqdv_slope_max", "dqdv_slope_min", "dqdv_min", "dqdv_avg",
    "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",
)
N_SUMMARY = len(SUMMARY_KEYS) + 1   # The scalars, plus the position feature.

# Fail at import time if the exporter stops producing a key consumed here.
# Left unchecked, the missing column would read as a constant zero and
# surface only as unexplained accuracy loss.
_MISSING = tuple(k for k in SUMMARY_KEYS if k not in SCALAR_KEYS)
if _MISSING:
    raise ImportError(
        f"gen_feature_bml_v2.SCALAR_KEYS no longer exports {_MISSING}; "
        f"update SUMMARY_KEYS or the exporter."
    )

# ---- L1: label ---------------------------------------------------------

def rul_to_class(rul: float) -> int:
    for cls, edge in enumerate(RUL_EDGES):
        if rul > edge:
            return cls
    return len(RUL_EDGES)

def window_rul(cycle_life: int, cycle_index: np.ndarray, start: int,
               n_random: int = N_RANDOM) -> int:
    end_cycle = int(cycle_index[start + n_random - 1])
    return max(0, cycle_life - end_cycle)

# ---- L2: io ------------------------------------------------------------

def _column(data, key: str, dtype=np.float32) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"{key!r} missing from .npz; found {sorted(data.files)}")
    return np.asarray(data[key], dtype=dtype).reshape(-1)

def load_cell_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as d:
        summary = {k: _column(d, NPZ_ALIAS.get(k, k)) for k in SCALAR_KEYS}
        cycle_index = _column(d, "cycle_index", np.int32)
        qdlin = np.asarray(d["qdlin"], dtype=np.float32)
        cycle_life = int(d["cycle_life"])
        eol_reached = bool(d["eol_reached"]) if "eol_reached" in d.files else None
        retention = float(d["retention"]) if "retention" in d.files else None

    n_cyc = summary["Qd"].size
    if qdlin.ndim != 2 or qdlin.shape != (n_cyc, V_BINS):
        raise ValueError(f"{path}: qdlin is {qdlin.shape}, expected ({n_cyc}, {V_BINS})")
    if cycle_index.size != n_cyc:
        raise ValueError(f"{path}: {cycle_index.size} cycle numbers for {n_cyc} cycles")
    if cycle_life != int(cycle_index[-1]):
        raise ValueError(
            f"{path}: cycle_life {cycle_life} != last cycle {int(cycle_index[-1])}; "
            f"the file was not truncated at end-of-life"
        )

    return {
        "cycle_life": cycle_life,
        "cycle_index": cycle_index,
        "summary": summary,
        "qdlin": qdlin,
        "eol_reached": eol_reached,
        "retention": retention,
    }

def load_all_npz(content_dir: str) -> dict:
    print("Loading BML .npz feature files...")
    cells: dict = {}
    root = Path(content_dir)

    for path in sorted(root.rglob("*.npz")):
        if path.name.startswith("._"):        # macOS AppleDouble sidecar
            continue
        cell_id = "__".join(path.relative_to(root).with_suffix("").parts)
        try:
            cells[cell_id] = load_cell_npz(str(path))
        except Exception as exc:
            print(f"  WARNING: could not load {path}: {exc}")

    n_aged = sum(1 for c in cells.values() if c["eol_reached"])
    print(f"  Loaded {len(cells)} BML cells "
          f"({n_aged} reached end-of-life, {len(cells) - n_aged} stopped short)")
    return cells

# ---- L3: index ---------------------------------------------------------

TAIL_MARGIN = 4
def window_starts_by_class(cell: dict, n_early: int = N_EARLY,
                           n_random: int = N_RANDOM) -> dict:
    """Sort valid random-window starts into RUL-class baskets.

    Each possible start after the early baseline is tested once:
    take the window starting there, compute the RUL at the window's last
    cycle, convert that RUL to a class, then store the start in that
    class's list.

    Args:
        cell: A cell dict from :func:`load_cell_npz`.
        n_early: Leading cycles carried by every sample.
        n_random: Consecutive cycles drawn from past the lead-in.

    Returns:
        {class_index: [window_start, ...]}; classes with no windows get [].
    """
    cycle_index, cycle_life = cell["cycle_index"], cell["cycle_life"]
    starts = {c: [] for c in range(N_CLASSES)}
    for start in range(n_early, cycle_index.size - n_random - TAIL_MARGIN + 1):
        cls = rul_to_class(window_rul(cycle_life, cycle_index, start, n_random))
        starts[cls].append(start)
    return starts

def build_sample_index(cells: dict, cell_ids: list, seed: int = 42,
                       n_early: int = N_EARLY, n_random: int = N_RANDOM) -> list:
    """Choose balanced sample windows from each cell.

    Returns sample addresses: (cell_id, window_start, label, rul).
    Each cell contributes the same number of samples per RUL class.
    Cells that are too short or missing any class are skipped.

    Args:
        cells: Mapping of cell id to cell dict.
        cell_ids: Which cells to draw from.
        seed: Base seed; each cell derives its own generator from it.
        n_early: Leading cycles carried by every sample. Overriding this
            (and n_random) is what a window-geometry grid search needs;
            both default to the module constants so existing callers are
            unaffected.
        n_random: Consecutive cycles drawn from past the lead-in.
    """
    index, dropped = [], []
    n_input = n_early + n_random

    for i, cid in enumerate(cell_ids):
        if cid not in cells:
            continue

        cell = cells[cid]

        # Need enough cycles to form one full sample window.
        if cell["cycle_index"].size < n_input:
            dropped.append((cid, f"only {cell['cycle_index'].size} cycles"))
            continue

        # Put all legal starts into class baskets.
        starts = window_starts_by_class(cell, n_early, n_random)

        # Balance this cell by sampling the rarest class count from every class.
        n_per_class = min(len(s) for s in starts.values())

        # If one basket is empty, this cell cannot be class-balanced.
        if n_per_class == 0:
            empty = [c for c, s in starts.items() if not s]
            dropped.append((cid, f"no window in class {empty}"))
            continue

        cell_rng = np.random.default_rng(seed + i + 3)

        for label, class_starts in starts.items():
            # Pick starts without duplicates.
            for start in cell_rng.choice(class_starts, size=n_per_class, replace=False):
                start = int(start)
                rul = window_rul(cell["cycle_life"], cell["cycle_index"], start, n_random)
                index.append((cid, start, label, rul))

    if dropped:
        print(f"  {len(dropped)} of {len(cell_ids)} cells contributed no samples:")
        for cid, why in dropped:
            print(f"    {cid}: {why}")

    return index

# ---- L4: featurise -----------------------------------------------------

class CellCache:
    """Per-cycle model inputs for one cell, computed once and reused.

    Only the summary matrix is materialised. The dq curves are derived from
    ``qdlin`` by one vectorised subtraction per sample, which measures 1.9x
    slower than slicing a prebuilt cache but avoids holding a second copy of
    every qdlin curve -- 194 MB across the 64 MATR cells measured, and again
    for every DataLoader worker.

    Attributes:
        summary: ``(n_cyc, N_SUMMARY)`` scalars plus the position feature.
        qdlin: The cell's ``(n_cyc, V_BINS)`` discharge curves.
        ref_qdlin: The curve every other cycle is measured against.
    """

    __slots__ = ("summary", "qdlin", "ref_qdlin")

    def __init__(self, cell: dict):
        """Build the summary matrix for one cell.

        Args:
            cell: A cell dict from :func:`load_cell_npz`.
        """
        scalars = np.stack([cell["summary"][k] for k in SUMMARY_KEYS], axis=1)
        # Real cycle numbers, not array positions: the two coincide on MATR
        # but diverge wherever logged cycle numbers skip.
        position = np.maximum(cell["cycle_index"], 1)[:, None] / POS_SCALE
        summary = np.concatenate([scalars, position], axis=1)
        # Non-finite scalars become zero, matching the exporter's own
        # convention for cycles it could not measure.
        self.summary = np.nan_to_num(summary.astype(np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0)

        self.qdlin = cell["qdlin"]
        self.ref_qdlin = cell["qdlin"][min(REF_CYCLE, len(cell["qdlin"]) - 1)]

    def window(self, start: int, n_early: int = N_EARLY,
              n_random: int = N_RANDOM) -> tuple:
        """Slice the cycles one sample is built from.

        Args:
            start: Array position where the random window begins.
            n_early: Leading cycles carried by every sample.
            n_random: Consecutive cycles drawn from past the lead-in.

        Returns:
            ``(dq, summary)`` of shapes ``(n_early + n_random, 1, V_BINS)``
            and ``(n_early + n_random, N_SUMMARY)``.
        """
        rows = np.concatenate([np.arange(n_early),
                               np.arange(start, start + n_random)])
        dq = (self.qdlin[rows] - self.ref_qdlin)[:, None, :]
        return dq, self.summary[rows]


def build_sample_tensors(cell: dict, start: int, dq_scaler, summary_scaler,
                         cache: CellCache = None, n_early: int = N_EARLY,
                         n_random: int = N_RANDOM) -> tuple:
    if cache is None:
        cache = CellCache(cell)
    dq_seq, summary_seq = cache.window(start, n_early, n_random)

    if dq_scaler is not None:
        T, C, F = dq_seq.shape
        dq_seq = dq_scaler.transform(dq_seq.reshape(T, C * F)).reshape(T, C, F)
        summary_seq = summary_scaler.transform(summary_seq)

    return (torch.tensor(dq_seq, dtype=torch.float32),
            torch.tensor(summary_seq, dtype=torch.float32))

# ---- L5: dataset -------------------------------------------------------


class BMLBatteryClsDataset(Dataset):
    """Windows of one cell group, featurised on demand.

    The sample index lives in RAM; signal data is sliced per ``__getitem__``
    from a per-cell cache built the first time that cell is touched.

    Args:
        cells: Mapping of cell id to cell dict.
        cell_ids: Which cells this split draws from.
        scalers: ``(dq_scaler, summary_scaler)`` fitted elsewhere, normally
            the training split's. Mutually exclusive with ``fit_scaler``.
        seed: Seed for window selection.
        fit_scaler: Fit fresh scalers from this split. Only the training
            split should do this; val and test must receive its scalers.
        n_early: Leading cycles carried by every sample.
        n_random: Consecutive cycles drawn from past the lead-in.
    """

    def __init__(self, cells: dict, cell_ids: list, scalers: tuple = None,
                 seed: int = 42, fit_scaler: bool = False,
                 n_early: int = N_EARLY, n_random: int = N_RANDOM):
        self.cells = cells
        self.n_early = n_early
        self.n_random = n_random
        self.index = build_sample_index(cells, cell_ids, seed=seed,
                                        n_early=n_early, n_random=n_random)
        self._cache: dict = {}

        if not self.index:
            raise ValueError("No valid samples.")

        print(f"  Total index entries: {len(self.index)}")
        labels = [e[2] for e in self.index]
        for c in range(N_CLASSES):
            print(f"  Class {c}: {labels.count(c)} samples")

        if scalers is not None:
            self.dq_scaler, self.summary_scaler = scalers
        elif fit_scaler:
            self.dq_scaler, self.summary_scaler = self._fit_scalers(seed)
        else:
            self.dq_scaler = self.summary_scaler = None

    def _cache_for(self, cell_id: str) -> CellCache:
        """Return this cell's cache, building it on first request."""
        cache = self._cache.get(cell_id)
        if cache is None:
            cache = self._cache[cell_id] = CellCache(self.cells[cell_id])
        return cache

    def _fit_scalers(self, seed: int, max_fit: int = 20000,
                     chunk: int = 500) -> tuple:
        """Fit both scalers from a random subset of this split's samples.

        Fitted incrementally: materialising all ``max_fit`` samples at once
        costs 1.28 GB as float32 and 2.56 GB once StandardScaler upcasts,
        whereas chunks of 500 peak at 32 MB. Measured over 6,000 samples,
        the two routes agree to 2.8e-17 on ``mean_`` and produce bit-wise
        identical ``transform`` output.

        Args:
            seed: Seed choosing which samples to fit on.
            max_fit: Cap on samples used for fitting.
            chunk: Samples materialised at once.

        Returns:
            ``(dq_scaler, summary_scaler)``.
        """
        print("  Fitting scalers on subset...")
        rng = np.random.default_rng(seed)
        subset = rng.choice(len(self.index),
                            size=min(max_fit, len(self.index)), replace=False)

        dq_scaler, summary_scaler = StandardScaler(), StandardScaler()
        for i in range(0, len(subset), chunk):
            dq_rows, sum_rows = [], []
            for k in subset[i:i + chunk]:
                cell_id, start, _, _ = self.index[k]
                dq, summary = self._cache_for(cell_id).window(
                    start, self.n_early, self.n_random)
                dq_rows.append(dq.reshape(self.n_early + self.n_random, -1))
                sum_rows.append(summary)
            dq_scaler.partial_fit(np.concatenate(dq_rows))
            summary_scaler.partial_fit(np.concatenate(sum_rows))

        print("  Scalers fitted.")
        return dq_scaler, summary_scaler

    def get_scalers(self) -> tuple:
        """Return ``(dq_scaler, summary_scaler)`` for reuse by other splits."""
        return self.dq_scaler, self.summary_scaler

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        cell_id, start, label, _ = self.index[idx]
        dq, summary = build_sample_tensors(
            self.cells[cell_id], start, self.dq_scaler, self.summary_scaler,
            cache=self._cache_for(cell_id),
            n_early=self.n_early, n_random=self.n_random)
        return {"dq": dq, "summary": summary,
                "label": torch.tensor(label, dtype=torch.long)}


def split_cell_ids(cell_ids, val_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Split cells three ways, by cell rather than by window.

    Splitting by cell is what keeps the evaluation honest: windows of one
    cell overlap heavily, so scattering them across splits would let the
    model meet the test cells during training.

    Args:
        cell_ids: All available cell ids.
        val_ratio: Share of cells going to validation, and again to test.
        seed: Seed for the shuffle.

    Returns:
        ``(train_ids, val_ids, test_ids)``.
    """
    ids = list(cell_ids)
    np.random.default_rng(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[2 * n_val:], ids[:n_val], ids[n_val:2 * n_val]


def _print_cell_names(name: str, ids: list, per_line: int = 6) -> None:
    """Print one split as a pasteable Python list literal."""
    lines = [ids[i:i + per_line] for i in range(0, len(ids), per_line)]
    inner = ",\n                ".join(
        ", ".join(f"'{n}'" for n in line) for line in lines)
    print(f"{name} = [{inner}]")


def build_clf_dataloaders(content_dir: str, batch_size: int = 32,
                          val_ratio: float = 0.2, num_workers: int = 0,
                          seed: int = 42, n_samples: int = None,
                          n_early: int = N_EARLY,
                          n_random: int = N_RANDOM) -> tuple:
    """Load a feature tree and return the three dataloaders and the scalers.

    Args:
        content_dir: Root of the exported .npz tree.
        batch_size: Samples per batch.
        val_ratio: Share of cells for validation, and again for test.
        num_workers: DataLoader worker processes.
        seed: Seed for the cell split and for window selection.
        n_samples: Accepted and ignored. Every cell already contributes as
            many windows per class as its rarest class allows, so there is
            no budget to set. Kept because the trainers still pass it.
        n_early: Leading cycles carried by every sample. Both default to
            the module constants, so callers that never pass them see no
            change; a window-geometry sweep is the reason to override them.
        n_random: Consecutive cycles drawn from past the lead-in.

    Returns:
        ``(train_loader, val_loader, test_loader, scalers)``.
    """
    cells = load_all_npz(content_dir)
    trn_ids, val_ids, test_ids = split_cell_ids(list(cells), val_ratio, seed)

    _print_cell_names("TrainCellName", trn_ids)
    _print_cell_names("ValidCellName", val_ids)
    _print_cell_names("TestCellName", test_ids)
    print(f"Split - Train: {len(trn_ids)}  Val: {len(val_ids)}   "
          f"Test: {len(test_ids)}")

    ds_kw = dict(n_early=n_early, n_random=n_random)
    print("Building train dataset...")
    train_ds = BMLBatteryClsDataset(cells, trn_ids, seed=seed, fit_scaler=True, **ds_kw)
    scalers = train_ds.get_scalers()
    print("Building val dataset...")
    val_ds = BMLBatteryClsDataset(cells, val_ids, scalers=scalers, seed=seed + 1, **ds_kw)
    print("Building test dataset...")
    test_ds = BMLBatteryClsDataset(cells, test_ids, scalers=scalers, seed=seed + 2, **ds_kw)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
            scalers)


def _describe_loader(name: str, loader: DataLoader) -> None:
    """Print one batch's shapes as a smoke test."""
    print(f"{name}: {len(loader.dataset)} samples")
    batch = next(iter(loader))
    for key in ("dq", "summary", "label"):
        print(f"  {key:8s} {tuple(batch[key].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate BML RUL classification dataloaders.")
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir=args.content_dir, batch_size=args.batch_size,
        val_ratio=args.val_ratio, num_workers=args.num_workers, seed=args.seed)

    print("\nDataloader check:")
    _describe_loader("Train", train_loader)
    _describe_loader("Val", val_loader)
    _describe_loader("Test", test_loader)
    print(f"Scaler features - dq: {scalers[0].n_features_in_}  "
          f"summary: {scalers[1].n_features_in_}")


if __name__ == "__main__":
    main()