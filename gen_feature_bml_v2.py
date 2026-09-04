"""Feature extraction for BatteryML cells (rewrite of gen_features_bml.py).

The pipeline is layered; each layer depends only on the one below it:

    L0  utils      pure array coercion and gap filling, no battery knowledge
    L1  segment    split one cycle into its charge / discharge parts
    L2  cycle      per-cycle feature extraction
    L3  cell       stack cycles, derive the voltage grid, truncate at EOL
    L4  io         file discovery and .npz export

One rule governs L0 and L1: coercion and gap filling are separate concerns.
`_as_array` preserves NaN so the finite masks in L1 can do real work; only
`_interp_nan`, applied to per-cycle summary series in L3, fills gaps. Filling
gaps in raw within-cycle series fabricates samples that pass the sign masks.
"""

from __future__ import annotations   # PEP 604 unions on Python 3.7+.

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

V_BINS = 1000        # Length of the resampled qdlin / dqdv curves.
MIN_CUR = 1e-1       # A. Dead band separating charge, discharge and rest.
EOL_FRACTION = 0.80  # Capacity retention that defines end-of-life.
EOL_CONFIRM = 10     # Cycles that must corroborate a threshold crossing.
EOL_CONFIRM_FRAC = 0.5   # Fraction of them that must stay below it.

_STD_EPS = 1e-9      # Floor on std before taking log, to avoid log(0).


# ---- L0: utils ---------------------------------------------------------


def _as_array(value: Any) -> np.ndarray:
    """Coerce a raw pickle field to a 1-D float32 array, preserving NaN/inf.

    Use for every within-cycle series (V, I, t, Q). Dropping bad samples is
    the job of the finite mask in `split_segment`, not of this function.
    `ravel` also normalises the 0-d scalars some fields carry (for example
    `internal_resistance_in_ohm`) into length-1 arrays.

    Returns an empty array when the field is missing or cannot be coerced.
    """
    if value is None:
        return np.zeros(0, dtype=np.float32)
    try:
        return np.asarray(value, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return np.zeros(0, dtype=np.float32)


def _interp_nan(arr: Any, fill: float = 0.0) -> np.ndarray:
    """Fill NaN/inf by linear interpolation over the array index.

    Intended for per-cycle summary series in L3, where an occasional failed
    cycle must be bridged without shifting the index alignment against
    `cycle_index`. Do not use on raw within-cycle series -- see `_as_array`.

    `np.interp` does not extrapolate, so leading and trailing gaps are clamped
    to the nearest valid value (forward/backward fill). If every element is
    non-finite the array is set to `fill`.
    """
    a = np.array(arr, dtype=np.float32).ravel()  # np.array always copies.
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        a[:] = fill
        return a
    idx = np.arange(a.size)
    a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Read a scalar metadata field, falling back to `default`.

    Pass `default=np.nan` when the caller needs to distinguish a missing
    field from a legitimate zero.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _log_std(arr: np.ndarray) -> float:
    """Standard deviation of a series in dB: 20*log10(std).

    Accumulates in float64; the caller normally supplies float32. Returns 0.0
    when fewer than two finite samples remain, which is indistinguishable
    from a real std of 1.0 -- behaviour retained from the original exporter.

    The value depends on sampling rate and series length, so it is only
    comparable within a single dataset.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(20.0 * np.log10(max(float(np.std(a)), _STD_EPS)))

# ---- L1: segment -------------------------------------------------------


class Segment(NamedTuple):
    """The charge or discharge part of one cycle, already filtered.

    All four arrays share a length and contain no non-finite values. Being a
    NamedTuple it is immutable, so a segment cannot be mutated in place once
    measured.

    Attributes:
        v:         Voltage, V.
        q:         Cumulative capacity, Ah.
        i:         Current, A.
        t:         Time, s, rebased so t[0] == 0.
        duration:  Time actually spent in this mode, excluding rest gaps.
        span:      t[-1] - t[0], which includes rest gaps. Equal to `duration`
                   only when `n_runs == 1`.
        n_runs:    Number of contiguous stretches. Multi-step charge protocols
                   produce more than one.
        n_dropped: Samples rejected as non-finite, for diagnostics.
    """

    v: np.ndarray
    q: np.ndarray
    i: np.ndarray
    t: np.ndarray
    duration: float
    span: float
    n_runs: int
    n_dropped: int

    @property
    def ok(self) -> bool:
        """True when the segment holds enough samples to featurise."""
        return self.t.size >= 2


def _empty_segment(n_dropped: int = 0) -> Segment:
    """Sentinel for a cycle with no usable segment.

    Deliberately empty rather than length-1 zeros: a zero-valued sample would
    reach the summary series as real data, whereas `ok == False` forces the
    caller to handle the failure.
    """
    z = np.zeros(0, dtype=np.float32)
    return Segment(v=z, q=z, i=z, t=z,
                duration=0.0, span=0.0, n_runs=0, n_dropped=n_dropped)


def split_segment(cycle: dict, kind: Literal["charge", "discharge"]) -> Segment:
    """Extract the charge or discharge segment of one cycle.

    Samples are classified by current sign with a `MIN_CUR` dead band, so rest
    periods and near-zero constant-voltage tails fall into neither segment.

    Args:
        cycle: One entry of the cell's `cycle_data` list.
        kind:  Which half of the cycle to return.

    Returns:
        A `Segment`; check `.ok` before use.
    """
    q_key = ("charge_capacity_in_Ah" if kind == "charge"
        else "discharge_capacity_in_Ah")
    v = _as_array(cycle.get("voltage_in_V"))
    q = _as_array(cycle.get(q_key))
    i = _as_array(cycle.get("current_in_A"))
    t = _as_array(cycle.get("time_in_s"))

    # Series within one cycle can disagree in length; trim to the shortest.
    n = min(v.size, q.size, i.size, t.size)
    if n < 2:
        return _empty_segment()
    v, q, i, t = v[:n], q[:n], i[:n], t[:n]
    # Two-stage mask: `finite` alone yields n_dropped, so bad samples stay
    # distinguishable from samples merely excluded by the sign test.
    finite = np.isfinite(v) & np.isfinite(q) & np.isfinite(i) & np.isfinite(t)
    mask = finite & (i > MIN_CUR if kind == "charge" else i < -MIN_CUR)
    n_dropped = int((~finite).sum())
    if int(mask.sum()) < 2:
        return _empty_segment(n_dropped)

    # Sum dt only across adjacent sample pairs that both belong to the
    # segment, which drops rest gaps. np.diff(t[mask]).sum() would instead
    # collapse to `span`.
    both = mask[:-1] & mask[1:]
    duration = float(np.diff(t)[both].sum())

    # Count False->True transitions; add one if the mask opens on True.
    n_runs = int((np.diff(mask.astype(np.int8)) == 1).sum()) + int(mask[0])

    ts = t[mask]
    return Segment(
        v=v[mask], q=q[mask], i=i[mask], t=ts - ts[0],
        duration=duration, span=float(ts[-1] - ts[0]),
        n_runs=n_runs, n_dropped=n_dropped,
    )


def charge_segment(cycle: dict) -> Segment:
    """Charge half of `cycle`."""
    return split_segment(cycle, "charge")


def discharge_segment(cycle: dict) -> Segment:
    """Discharge half of `cycle`."""
    return split_segment(cycle, "discharge")


# ---- L2: cycle ---------------------------------------------------------

# Single source of truth for the per-cycle scalars. L3 iterates this tuple to
# stack and export, so a new feature means one entry here plus one assignment
# in `cycle_features`.
SCALAR_KEYS = (
    "Qd", "Qc", "c_t", "dc_t",
    "dqdv_slope_max", "dqdv_slope_min", "dqdv_min", "dqdv_avg",
    "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic",
)


def _dedupe_interp(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample y(x) onto `grid`, averaging samples that share an x value.

    Used to turn the raw (voltage, capacity) pairs of one discharge into a
    fixed-length curve on a common voltage grid, so curves from different
    cycles can be compared element-wise.

    Args:
        x:    Independent variable, any order (discharge voltage is decreasing).
        y:    Dependent variable, same length as `x`.
        grid: Strictly increasing query points; defines the output length.

    Returns:
        `y` evaluated on `grid`, float32. All-zeros when `x` carries fewer
        than two distinct usable values.

    Notes:
        Callers in this module pass filtered `Segment` arrays, so the finite
        mask below is defensive only. `np.interp` does not extrapolate:
        queries outside the sampled range are clamped to the end values.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    # np.interp requires xp to be increasing; discharge voltage decreases.
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Collapse duplicate x values to their mean y. Duplicates are common on
    # voltage plateaus and wherever the logger quantises its voltage reading.
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    ys = np.zeros(ux.size, dtype=np.float64)
    cnt = np.zeros(ux.size, dtype=np.float64)
    np.add.at(ys, inv, y)          # np.add.at accumulates on repeated indices;
    np.add.at(cnt, inv, 1.0)       # ys[inv] += y would keep only one write.
    return np.interp(grid, ux, ys / np.maximum(cnt, 1.0)).astype(np.float32)


def cycle_number(cycle: dict, idx: int, already_spent: int) -> int:
    """Absolute cycle number, offset by cycles run before logging began.

    A recorded number no larger than `idx + 2` indicates a file that restarts
    its numbering at one, so `already_spent` is added back. Files that already
    number cycles absolutely are left alone.
    """
    try:
        n = int(cycle.get("cycle_number", idx + 1))
    except (TypeError, ValueError):
        n = idx + 1
    if already_spent > 0 and n <= idx + 2:
        return already_spent + n
    return n


def cycle_features(cycle: dict, voltage_grid: np.ndarray,
                   idx: int, already_spent: int = 0,
                   use_duration: bool = False) -> dict:
    """Extract every feature of a single cycle. Never raises.

    Args:
        cycle:         One entry of the cell's `cycle_data` list.
        voltage_grid:  Shared grid for this cell, from L3. Must be identical
                       across cycles, or the qdlin difference that forms the
                       model input is meaningless.
        idx:           Position of the cycle in the cell's list.
        already_spent: Cycles run before logging began.
        use_duration:  Take charge/discharge times from `Segment.duration`
                       rather than `.span`. `duration` excludes rest gaps and
                       is the truer measure, but it changes exported values,
                       so it requires regenerating features and retraining.

    Returns:
        A dict holding every key in `SCALAR_KEYS` plus `cycle_index`, `qdlin`
        and `dqdv`. Scalars that could not be computed are NaN so that L3 can
        interpolate them from neighbouring cycles; curves fall back to zeros.
        Charge and discharge fail independently.
    """
    dis = discharge_segment(cycle)
    ch  = charge_segment(cycle)

    out = {k: np.nan for k in SCALAR_KEYS}
    out["cycle_index"] = cycle_number(cycle, idx, already_spent)
    out["qdlin"] = np.zeros(V_BINS, dtype=np.float32)
    out["dqdv"]  = np.zeros(V_BINS, dtype=np.float32)

    if not dis.ok:
        return out  # Unusable cycle: NaN scalars let the caller interpolate.

    qdlin = _dedupe_interp(dis.v, dis.q, voltage_grid)
    dqdv  = _interp_nan(np.gradient(qdlin, voltage_grid).astype(np.float32))

    # Slope features: trim the noisy grid edges, then smooth before diffing.
    w  = np.convolve(dqdv[100:900], np.ones(10) / 10, mode="valid")
    sl = np.diff(w)

    out.update(
        qdlin = qdlin,
        dqdv  = dqdv,
        Qd    = float(np.nanmax(dis.q)),
        dc_t  = float(dis.duration if use_duration else dis.span),
        dqdv_slope_max = float(np.max(sl)),
        dqdv_slope_min = float(np.min(sl)),
        dqdv_min = float(np.nanmin(dqdv)),
        dqdv_avg = float(np.nanmean(dqdv)),
        log_std_Qd = _log_std(dis.q),
        log_std_Id = _log_std(dis.i),
    )

    if ch.ok:
        out.update(
            Qc   = float(np.nanmax(ch.q)),
            c_t  = float(ch.duration if use_duration else ch.span),
            log_std_Qc = _log_std(ch.q),
            log_std_Ic = _log_std(ch.i),
        )
    return out


# ---- L3: cell ----------------------------------------------------------


def voltage_limits(cell: dict, cycles: list) -> tuple:
    """Voltage span used to build this cell's resampling grid.

    Prefers the limits declared in the cell metadata. Falls back to the 1st
    and 99th percentile of observed discharge voltage over the first fifty
    cycles: percentiles rather than min/max so that one stray sample cannot
    stretch the grid and squeeze every real point into a narrow band, and
    early cycles because the grid must be fixed for the whole cell and a
    healthy cell spans the widest range.

    Returns:
        (vmin, vmax), falling back to (0.0, 1.0) when nothing usable is found.
    """
    vmin = _safe_float(cell.get("min_voltage_limit_in_V"), np.nan)
    vmax = _safe_float(cell.get("max_voltage_limit_in_V"), np.nan)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        return vmin, vmax

    samples = []
    for cyc in cycles[:50]:
        seg = discharge_segment(cyc)
        if seg.ok:
            samples.append(seg.v)
            continue
        v = _as_array(cyc.get("voltage_in_V"))   # Salvage: any finite voltage.
        if v.size:
            samples.append(v[np.isfinite(v)])

    if not samples:
        return 0.0, 1.0
    allv = np.concatenate(samples)
    if allv.size == 0:
        return 0.0, 1.0

    lo = float(np.nanpercentile(allv, 1))
    hi = float(np.nanpercentile(allv, 99))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _smooth_edge(y: np.ndarray, k: int) -> np.ndarray:
    """Moving average that replicates boundary values instead of zero-padding.

    np.convolve pads with zeros, which drags the smoothed head and tail toward
    zero and makes a subsequent argmin latch onto the array edges.
    """
    if k < 2 or y.size < 2:
        return y
    k = min(k, y.size)
    lo = k // 2
    hi = k - 1 - lo
    return np.convolve(np.pad(y, (lo, hi), mode="edge"),
                       np.ones(k) / k, mode="valid")


def find_eol_idx(qd: np.ndarray, eol_fraction: float = EOL_FRACTION,
                 confirm: int = EOL_CONFIRM,
                 confirm_frac: float = EOL_CONFIRM_FRAC) -> int:
    """Index of the end-of-life cycle.

    Q_initial is the maximum over the first ten usable cycles, which skips
    formation anomalies and early dips. A cycle counts as the crossing only
    when at least `confirm_frac` of the following `confirm` cycles also sit
    below `eol_fraction * Q_initial`. Capacity is not monotonic -- logging
    glitches and post-rest capacity recovery both produce isolated cycles that
    dip under the threshold and return -- so a bare first crossing is not a
    reliable endpoint.

    Cells that never fade that far fall back to their most-aged cycle,
    smoothed so a single noisy dip cannot win. Inspect `CellFeatures.retention`
    to tell the two cases apart.

    Pass `confirm=0` to recover the original first-crossing rule.

    Returns:
        The index, or `len(qd)` when no endpoint can be determined.
    """
    qd = np.asarray(qd, dtype=np.float32)
    valid = np.isfinite(qd) & (qd > 0)
    if valid.sum() < 2:
        return len(qd)

    vi = np.where(valid)[0]
    n_ref = min(10, vi.size)
    q_init = float(np.max(qd[vi[:n_ref]]))
    if q_init <= 0:
        return len(qd)

    post = vi[vi > vi[n_ref - 1]]
    if post.size == 0:
        return len(qd)

    # Rolling fraction of sub-threshold cycles over [k, k+confirm], via prefix
    # sums. The window shrinks near the tail rather than disqualifying it.
    below = (qd[post] < eol_fraction * q_init).astype(np.float64)
    start = np.arange(below.size)
    end = np.minimum(start + confirm + 1, below.size)
    csum = np.concatenate([[0.0], np.cumsum(below)])
    frac = (csum[end] - csum[start]) / (end - start)

    confirmed = np.where((below > 0) & (frac >= confirm_frac))[0]
    if confirmed.size:
        return int(post[confirmed[0]])
    return int(post[np.argmin(_smooth_edge(qd[post], confirm))])


class CellFeatures(NamedTuple):
    """Per-cell feature arrays, already truncated at end-of-life.

    Attributes:
        arrays:      Every SCALAR_KEYS series plus cycle_index, qdlin and
                     dqdv. All share length n_keep and hold no NaN.
        cycle_life:  Absolute cycle number of the end-of-life cycle.
        retention:   Capacity at that cycle relative to Q_initial.
        eol_reached: True when the cell actually faded past EOL_FRACTION.
                     False means `cycle_life` came from the min-capacity
                     fallback and understates the true life, so the RUL labels
                     derived from it are compressed.
    """

    arrays: dict
    cycle_life: int
    retention: float
    eol_reached: bool


def cell_features(cell: dict, use_duration: bool = False,
                  confirm: int = EOL_CONFIRM) -> CellFeatures | None:
    """Featurise every cycle of one cell and truncate at end-of-life.

    Deliberately reports rather than enforces: a cell whose logging stopped
    short of end-of-life still returns full arrays, with `eol_reached` False.
    Whether to keep such a cell is L4's decision, which keeps the policy in
    one readable place and keeps this function testable.

    Args:
        cell:         Unpickled cell dict.
        use_duration: Forwarded to `cycle_features`.
        confirm:      Forwarded to `find_eol_idx`; 0 restores the original
                      first-crossing rule, which is how legacy labels are
                      reproduced for an A/B comparison.

    Returns:
        A `CellFeatures`, or None when the cell holds no usable cycle.
    """
    cycles = [c for c in cell.get("cycle_data", []) if isinstance(c, dict)]
    vmin, vmax = voltage_limits(cell, cycles)
    grid = np.linspace(vmin, vmax, V_BINS, dtype=np.float32)
    spent = int(cell.get("already_spent_cycles") or 0)

    rows = [cycle_features(c, grid, i, spent, use_duration)
            for i, c in enumerate(cycles)]
    if not rows:
        return None

    # SCALAR_KEYS drives the stacking, so adding a feature touches one place.
    # _interp_nan bridges cycles that failed in L2 without breaking the
    # index alignment against cycle_index.
    arr = {k: _interp_nan([r[k] for r in rows]) for k in SCALAR_KEYS}
    arr["cycle_index"] = np.asarray([r["cycle_index"] for r in rows], dtype=np.int32)
    arr["qdlin"] = np.stack([r["qdlin"] for r in rows]).astype(np.float32)
    arr["dqdv"]  = np.stack([r["dqdv"]  for r in rows]).astype(np.float32)

    # EOL must be found after gap filling: a NaN Qd fails the `qd > 0` test
    # and would shift the endpoint.
    qd = arr["Qd"]
    eol = find_eol_idx(qd, confirm=confirm)
    n_keep = min(eol + 1, qd.size)          # Keep the EOL cycle itself.
    q_init = float(np.max(qd[:min(10, qd.size)]))

    if eol < qd.size:
        life = int(arr["cycle_index"][eol])
        retention = float(qd[eol] / q_init) if q_init > 0 else 0.0
    else:
        life = int(arr["cycle_index"][-1])
        retention = 0.0

    return CellFeatures({k: a[:n_keep] for k, a in arr.items()},
                        life, retention, retention < EOL_FRACTION)

# ---- L4: io ------------------------------------------------------------

MAX_RETENTION = 0.85  # Reject cells still above this at their last cycle.

# Directories that hold label JSON rather than cell pickles.
LABEL_DIR_NAMES = ("Life labels", "Life labels 2", "READMEs")

# SCALAR_KEYS use the in-code capitalisation; the .npz keeps the lower-case
# names the downstream loader already reads. Public because the loader
# imports it to translate back -- renaming a key here must break that import
# rather than silently yield a missing column downstream.
NPZ_ALIAS = {"Qd": "qd", "Qc": "qc"}


def iter_pkl_files(data_dir: Path) -> list[Path]:
    """Every cell pickle under `data_dir`, sorted for reproducible runs.

    Skips the label directories and the AppleDouble `._` sidecars that macOS
    leaves on shared drives.
    """
    skipped = set(LABEL_DIR_NAMES)
    paths = []
    for path in data_dir.rglob("*.pkl"):
        if path.name.startswith("._"):
            continue
        if set(path.relative_to(data_dir).parts[:-1]) & skipped:
            continue
        paths.append(path)
    return sorted(paths)


def should_skip(cf: CellFeatures, max_retention: float = MAX_RETENTION) -> bool:
    """True for cells whose logging stopped well short of end-of-life.

    Their `cycle_life` is the last logged cycle rather than a real endpoint,
    so every RUL label derived from them is compressed. Cells that merely
    missed the threshold by a little are kept: dropping them costs far more
    data than the label noise they carry -- 81 of 169 MATR cells fall in that
    band. Inspect `eol_reached` downstream instead.
    """
    return not cf.eol_reached and cf.retention > max_retention


def extract_pkl(path: Path, data_dir: Path, out_dir: Path,
                use_duration: bool = False,
                confirm: int = EOL_CONFIRM,
                max_retention: float = MAX_RETENTION) -> str:
    """Featurise one pickle and write the .npz mirroring its source layout.

    Returns:
        A short ASCII status token: "ok", or one of the "skip:*" reasons.
        Callers tally these instead of parsing log lines.
    """
    if path.stat().st_size == 0:
        return "skip:empty_file"

    with path.open("rb") as f:
        cell = pickle.load(f)
    if not isinstance(cell, dict):
        return "skip:not_a_dict"

    cf = cell_features(cell, use_duration=use_duration, confirm=confirm)
    if cf is None:
        return "skip:no_usable_cycle"
    if should_skip(cf, max_retention):
        return "skip:not_aged"

    cell_id = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                     str(cell.get("cell_id") or path.stem))
    out_sub = out_dir / path.parent.relative_to(data_dir)
    out_sub.mkdir(parents=True, exist_ok=True)

    arrays = cf.arrays
    dummy = np.zeros(arrays["Qd"].size, dtype=np.float32)

    payload = {NPZ_ALIAS.get(k, k): arrays[k] for k in SCALAR_KEYS}
    payload.update(
        source_file=str(path),
        cell_id=cell_id,
        cycle_life=np.array(cf.cycle_life, dtype=np.int32),
        cycle_index=arrays["cycle_index"],
        qdlin=arrays["qdlin"],
        dqdv=arrays["dqdv"],
        # Retained for backward compatibility with the original exporter.
        # BatteryML supplies neither consistently, so they stay placeholders.
        IR=dummy, tmax=dummy, tavg=dummy,
        # Label-quality flags. Unknown keys are ignored by the loader, so
        # adding them does not invalidate existing checkpoints.
        eol_reached=np.array(cf.eol_reached),
        retention=np.array(cf.retention, dtype=np.float32),
    )
    np.savez_compressed(out_sub / f"{cell_id}.npz", **payload)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BatteryML .pkl cells to .npz feature files.")
    parser.add_argument("--data_dir", default="./Raw/Raw_BML")
    parser.add_argument("--out_dir", default="./content_bml")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Smoke-test limit on the number of cells.")
    parser.add_argument("--use_duration", action="store_true",
                        help="Charge/discharge times exclude rest gaps. "
                             "Changes exported values; requires retraining.")
    parser.add_argument("--eol_confirm", type=int, default=EOL_CONFIRM,
                        help="Cycles corroborating an EOL crossing; "
                             "0 restores the first-crossing rule.")
    parser.add_argument("--max_retention", type=float, default=MAX_RETENTION,
                        help="Reject cells still above this retention.")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_pkl_files(data_dir)
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f"Found {len(files)} .pkl files under {data_dir}")

    # Log output stays ASCII: Windows consoles default to cp1252 and raise
    # UnicodeEncodeError on anything else.
    tally = Counter()
    for n, path in enumerate(files, start=1):
        status = extract_pkl(path, data_dir, out_dir,
                             use_duration=args.use_duration,
                             confirm=args.eol_confirm,
                             max_retention=args.max_retention)
        tally[status] += 1
        print(f"[{n}/{len(files)}] {status:20s} {path.name}")

    print(f"\nDone. Output: {out_dir}")
    for status, count in sorted(tally.items()):
        print(f"  {status:20s} {count}")


if __name__ == "__main__":
    main()