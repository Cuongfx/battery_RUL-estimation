"""summarize_results.py - Build Accuracy summary tables from checkpoint Info_log.txt files.

Reads every checkpoints_clf_bml_* directory produced by the RunToTrain scripts
and writes a Markdown report of test Accuracy, plus a long-format CSV of every
individual run, so partially-finished sweeps still produce a usable report.

Covers three experiment shapes:
    baseline      checkpoints_clf_bml_<DATASET>[_tf|_es]/Info_log.txt
                  (run_train_all_default.ps1 / run_train_all_model.ps1)
    exp15         checkpoints_clf_bml_EXP_<DATASET>/E_R_<early>_<random>_run<r>/
                  (run_exp_15.ps1 -- 5 n_early x 5 n_random, 5 repeats)
    model_adjust  checkpoints_clf_bml_ModelAdjust_<DATASET>/CNN_GRUd_GRUl_<cnn>_<grud>_<grul>_run<r>/
                  (run_model_adjust.ps1 -- 4 cnn_dim x 4 gru_dim x 3 gru_layers, 3 repeats)

Usage:
    python summarize_results.py --root . --out document/experiment_results.md --csv document/experiment_results.csv
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

ACC_RE = re.compile(r"Test Loss:\s*[\d.]+\s*Accuracy:\s*([\d.]+)")

DATASETS = ["HUST", "MATR", "LFP"]
BASELINE_VARIANTS = [("", "default (CNN+GRU)"), ("_tf", "transformer"), ("_es", "CMA-ES")]

E_VALUES = [2, 4, 6, 8, 10]
R_VALUES = [2, 4, 6, 8, 10]
N_RUNS_EXP = 5

CNN_VALUES = [8, 16, 32, 64]
GRU_VALUES = [8, 16, 32, 64]
LAYER_VALUES = [1, 2, 3]
N_RUNS_ADJ = 3


def read_accuracy(info_log_path: Path):
    """Last 'Test Loss: ... Accuracy: X' line in the log, as a percentage."""
    if not info_log_path.exists():
        return None
    text = info_log_path.read_text(encoding="utf-8", errors="ignore")
    matches = ACC_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1]) * 100


def fmt(v):
    return f"{v:.2f}" if v is not None else "-"


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def grid_table(row_label, col_label, rows, cols, get_value):
    lines = [f"| {row_label}\\{col_label} | " + " | ".join(str(c) for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        vals = [fmt(get_value(r, c)) for c in cols]
        lines.append(f"| {r} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ---- readers ---------------------------------------------------------------

def collect_baseline(root):
    """dataset -> variant -> accuracy"""
    out = {}
    for dataset in DATASETS:
        out[dataset] = {}
        for suffix, label in BASELINE_VARIANTS:
            path = root / f"checkpoints_clf_bml_{dataset}{suffix}" / "Info_log.txt"
            out[dataset][label] = read_accuracy(path)
    return out


def collect_exp15(root, dataset):
    """run -> e -> r -> accuracy"""
    base = root / f"checkpoints_clf_bml_EXP_{dataset}"
    data = {}
    for run in range(1, N_RUNS_EXP + 1):
        data[run] = {}
        for e in E_VALUES:
            data[run][e] = {}
            for r in R_VALUES:
                path = base / f"E_R_{e}_{r}_run{run}" / "Info_log.txt"
                data[run][e][r] = read_accuracy(path)
    return data


def collect_model_adjust(root, dataset):
    """run -> gru_layers -> cnn_dim -> gru_dim -> accuracy"""
    base = root / f"checkpoints_clf_bml_ModelAdjust_{dataset}"
    data = {}
    for run in range(1, N_RUNS_ADJ + 1):
        data[run] = {}
        for layer in LAYER_VALUES:
            data[run][layer] = {}
            for cnn in CNN_VALUES:
                data[run][layer][cnn] = {}
                for grud in GRU_VALUES:
                    path = base / f"CNN_GRUd_GRUl_{cnn}_{grud}_{layer}_run{run}" / "Info_log.txt"
                    data[run][layer][cnn][grud] = read_accuracy(path)
    return data


# ---- report ------------------------------------------------------------

def render_baseline(baseline):
    lines = ["## 1. Baseline runs (one training run per dataset)", ""]
    lines.append("Default hyperparameters (`n_early=10`, `n_random=10`, `cnn_dim=32`, `gru_dim=32`, "
                  "`gru_layers=2`), one run per dataset/script variant. Source: "
                  "`run_train_all_default.ps1` / `run_train_all_model.ps1`.")
    lines.append("")
    variants = [label for _, label in BASELINE_VARIANTS]
    lines.append("| Dataset | " + " | ".join(variants) + " |")
    lines.append("|" + "---|" * (len(variants) + 1))
    for dataset in DATASETS:
        row = [fmt(baseline[dataset][label]) for label in variants]
        lines.append(f"| {dataset} | " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_exp15(root):
    lines = ["## 2. EXP_15 grid search: N_EARLY x N_RANDOM", ""]
    lines.append("Source: `run_exp_15.ps1`. For each dataset, every combination of "
                  "`n_early` in {2,4,6,8,10} and `n_random` in {2,4,6,8,10} is trained "
                  "5 times (125 runs/dataset). Each cell is test Accuracy (%); row = "
                  "`n_early` (E), column = `n_random` (R).")
    lines.append("")
    for dataset in DATASETS:
        data = collect_exp15(root, dataset)
        lines.append(f"### {dataset}")
        lines.append("")
        for run in range(1, N_RUNS_EXP + 1):
            lines.append(f"**Run {run}**")
            lines.append("")
            lines.append(grid_table("E", "R", E_VALUES, R_VALUES,
                                     lambda e, r, d=data[run]: d[e][r]))
            lines.append("")

        def cell_values(e, r):
            return [data[run][e][r] for run in range(1, N_RUNS_EXP + 1)]

        lines.append("**Mean (5 runs)**")
        lines.append("")
        lines.append(grid_table("E", "R", E_VALUES, R_VALUES,
                                 lambda e, r: mean_std(cell_values(e, r))[0]))
        lines.append("")
        lines.append("**Standard deviation (5 runs)**")
        lines.append("")
        lines.append(grid_table("E", "R", E_VALUES, R_VALUES,
                                 lambda e, r: mean_std(cell_values(e, r))[1]))
        lines.append("")
    return "\n".join(lines)


def render_model_adjust(root):
    lines = ["## 3. ModelAdjust grid search: cnn_dim x gru_dim x gru_layers", ""]
    lines.append("Source: `run_model_adjust.ps1`. For each dataset, every combination of "
                  "`cnn_dim`/`gru_dim` in {8,16,32,64} and `gru_layers` in {1,2,3} is trained "
                  "3 times (144 runs/dataset). Because this is a 3D sweep, one "
                  "`cnn_dim x gru_dim` table is shown per `gru_layers` value; individual-run "
                  "tables are folded into Mean/Standard (raw per-run numbers are in the CSV).")
    lines.append("")
    for dataset in DATASETS:
        data = collect_model_adjust(root, dataset)
        lines.append(f"### {dataset}")
        lines.append("")
        for layer in LAYER_VALUES:

            def cell_values(cnn, grud, layer=layer):
                return [data[run][layer][cnn][grud] for run in range(1, N_RUNS_ADJ + 1)]

            lines.append(f"**gru_layers = {layer} - Mean ({N_RUNS_ADJ} runs)**")
            lines.append("")
            lines.append(grid_table("cnn_dim", "gru_dim", CNN_VALUES, GRU_VALUES,
                                     lambda cnn, grud, layer=layer: mean_std(cell_values(cnn, grud))[0]))
            lines.append("")
            lines.append(f"**gru_layers = {layer} - Standard deviation ({N_RUNS_ADJ} runs)**")
            lines.append("")
            lines.append(grid_table("cnn_dim", "gru_dim", CNN_VALUES, GRU_VALUES,
                                     lambda cnn, grud, layer=layer: mean_std(cell_values(cnn, grud))[1]))
            lines.append("")
    return "\n".join(lines)


def write_csv(root, csv_path):
    rows = [("experiment", "dataset", "variant", "n_early", "n_random",
             "cnn_dim", "gru_dim", "gru_layers", "run", "accuracy_pct")]

    baseline = collect_baseline(root)
    for dataset in DATASETS:
        for suffix, label in BASELINE_VARIANTS:
            acc = baseline[dataset][label]
            rows.append(("baseline", dataset, label, "", "", "", "", "", 1, fmt(acc)))

    for dataset in DATASETS:
        data = collect_exp15(root, dataset)
        for run in range(1, N_RUNS_EXP + 1):
            for e in E_VALUES:
                for r in R_VALUES:
                    rows.append(("exp15", dataset, "", e, r, "", "", "", run,
                                 fmt(data[run][e][r])))

    for dataset in DATASETS:
        data = collect_model_adjust(root, dataset)
        for run in range(1, N_RUNS_ADJ + 1):
            for layer in LAYER_VALUES:
                for cnn in CNN_VALUES:
                    for grud in GRU_VALUES:
                        rows.append(("model_adjust", dataset, "", "", "", cnn, grud, layer,
                                     run, fmt(data[run][layer][cnn][grud])))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root containing checkpoints_clf_bml_* dirs")
    parser.add_argument("--out", default="document/experiment_results.md")
    parser.add_argument("--csv", default="document/experiment_results.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    csv_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = collect_baseline(root)

    parts = [
        "# Experiment Results",
        "",
        "Auto-generated from `checkpoints_clf_bml_*/Info_log.txt` by `summarize_results.py`. "
        "All values are test-set Accuracy in %. Missing cells (`-`) mean that run's checkpoint "
        "directory or `Info_log.txt` was not found.",
        "",
        render_baseline(baseline),
        render_exp15(root),
        render_model_adjust(root),
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")
    write_csv(root, csv_path)
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
