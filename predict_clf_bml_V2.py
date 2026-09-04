"""Batch inference and reporting for a trained BML RUL classifier.

Script form of predict_clf_bml_V2.ipynb: slides the input window across
every cell's whole life, then writes the five figures, a plain-text report
and a markdown report into ./predict_bml_<name>/.

Both architectures are supported. Which one a checkpoint holds is read
from its own model_config.json rather than guessed: the transformer
trainer records d_model, the CNN+GRU trainer records gru_dim.

Usage:
    python predict_clf_bml_V2.py --ckpt_dir ./checkpoints_clf_bml_HUST
    python predict_clf_bml_V2.py --ckpt_dir ./checkpoints_clf_bml_HUST_es \
        --data_dir ./content_bml/HUST
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # cluster nodes have no display

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from dataset_clf_bml_v2 import (CellCache, N_CLASSES, N_EARLY, N_RANDOM,
                                RUL_EDGES, build_sample_tensors, load_all_npz,
                                rul_to_class, window_rul)
from model_clf import BatteryRULClassifier

CLASS_COLORS = ["#1565C0", "#2E7D32", "#F9A825", "#E65100", "#B71C1C"]


# ---- L1: naming --------------------------------------------------------

def class_names(edges=RUL_EDGES):
    """Readable name per class, derived from the dataset's own thresholds."""
    names = [f"RUL>{edges[0]}"]
    for i in range(len(edges) - 1):
        names.append(f"{edges[i + 1]}<RUL<={edges[i]}")
    names.append(f"RUL<={edges[-1]}")
    return names


def run_name(ckpt_dir):
    """'./checkpoints_clf_bml_HUST_es/' -> 'HUST_es'."""
    base = os.path.basename(os.path.normpath(ckpt_dir))
    prefix = "checkpoints_clf_bml_"
    return base[len(prefix):] if base.startswith(prefix) else base


# ---- L2: loading -------------------------------------------------------

def load_model(ckpt_dir, device):
    """Rebuild the architecture recorded next to the weights and load them.

    Returns:
        (model, cfg, arch_name)
    """
    cfg_path = os.path.join(ckpt_dir, "model_config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"ERROR: {cfg_path} not found - is this a checkpoint dir?")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if "d_model" in cfg:
        from train_clf_bml_transformer_V2 import BatteryRULTransformer
        model = BatteryRULTransformer(
            cnn_dim=cfg["cnn_dim"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_enc_layers=cfg["n_enc_layers"],
            d_ff=cfg["d_ff"], summary_feats=cfg["summary_feats"],
            n_classes=cfg["n_classes"], dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
        )
        arch = "CNN + Transformer"
    else:
        model = BatteryRULClassifier(
            cnn_dim=cfg["cnn_dim"], gru_dim=cfg["gru_dim"],
            gru_layers=cfg["gru_layers"], summary_feats=cfg["summary_feats"],
            n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        )
        arch = "CNN + BiGRU"

    weights = os.path.join(ckpt_dir, "best_clf_bml.pt")
    if not os.path.isfile(weights):
        sys.exit(f"ERROR: {weights} not found - training may not have saved yet.")
    model.load_state_dict(torch.load(weights, map_location=device,
                                     weights_only=True))
    return model.to(device).eval(), cfg, arch


def resolve_data_dir(args, cfg):
    """--data_dir if given, else the content_dir the trainer recorded."""
    if args.data_dir:
        return args.data_dir
    recorded = cfg.get("content_dir")
    if recorded and os.path.isdir(recorded):
        return recorded
    sys.exit("ERROR: could not locate the dataset. Pass --data_dir "
             f"(model_config.json recorded {recorded!r}, which is not a directory).")


# ---- L3: inference -----------------------------------------------------

@torch.no_grad()
def predict_cell(cell, model, scalers, device, n_early, n_random):
    """Slide the input window across a cell's whole life.

    n_early/n_random come from the checkpoint's own config, so a model
    trained by run_exp_15 with a non-default window is scored with that
    same window rather than the module defaults.
    """
    cycle_life = int(cell["cycle_life"])
    cycle_index = cell["cycle_index"]
    n_cyc = cycle_index.size
    cache = CellCache(cell)
    dq_sc, sum_sc = scalers

    end_cycles, pred_classes, true_classes, pred_probs = [], [], [], []

    for start in range(n_early, n_cyc - n_random + 1):
        dq_seq, sum_seq = build_sample_tensors(cell, start, dq_sc, sum_sc,
                                               cache=cache, n_early=n_early,
                                               n_random=n_random)
        logits = model(dq_seq.unsqueeze(0).to(device),
                       sum_seq.unsqueeze(0).to(device)).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        rul = window_rul(cycle_life, cycle_index, start, n_random=n_random)

        end_cycles.append(int(cycle_index[start + n_random - 1]))
        pred_classes.append(int(probs.argmax()))
        true_classes.append(rul_to_class(rul))
        pred_probs.append(probs)

    pred_classes = np.array(pred_classes)
    true_classes = np.array(true_classes)
    return {
        "end_cycles": np.array(end_cycles),
        "pred_classes": pred_classes,
        "true_classes": true_classes,
        "pred_probs": np.array(pred_probs),
        "cycle_life": cycle_life,
        "accuracy": float((pred_classes == true_classes).mean()),
    }


def neol_from_result(r):
    """First window-end cycle in the most-degraded class, true and predicted.

    The predicted side prefers a genuine (LAST-1 -> LAST) transition and
    only falls back to the first bare hit, so a single early misfire does
    not decide the estimate on its own.
    """
    last = N_CLASSES - 1
    cyc, p, t = r["end_cycles"], r["pred_classes"], r["true_classes"]

    hits_t = cyc[t == last]
    eol_t = int(hits_t[0]) if len(hits_t) else r["cycle_life"]

    trans = np.where((p[:-1] == last - 1) & (p[1:] == last))[0]
    if len(trans):
        eol_p = int(cyc[trans[0] + 1])
    else:
        hits_p = cyc[p == last]
        eol_p = int(hits_p[0]) if len(hits_p) else int(cyc[-1])
    return eol_t, eol_p


# ---- L4: figures -------------------------------------------------------

def plot_single_cell(cid, r, names, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [2, 2, 1.2]})
    fig.suptitle(f"RUL Classification - {cid}  "
                 f"(cycle life {r['cycle_life']}, accuracy {r['accuracy']:.2%})",
                 fontsize=13, fontweight="bold")

    cycles, pred_cls = r["end_cycles"], r["pred_classes"]
    true_cls, probs = r["true_classes"], r["pred_probs"]
    correct = pred_cls == true_cls

    ax = axes[0]
    ax.step(cycles, true_cls, where="post", color="#1565C0", lw=2.5,
            label="True class", zorder=3)
    ax.step(cycles, pred_cls, where="post", color="#E53935", lw=1.8, ls="--",
            label="Predicted class", zorder=4)
    ax.fill_between(cycles, -0.4, N_CLASSES - 0.6, where=correct,
                    alpha=0.08, color="green", step="post")
    ax.fill_between(cycles, -0.4, N_CLASSES - 0.6, where=~correct,
                    alpha=0.12, color="red", step="post")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(-0.4, N_CLASSES - 0.6)
    ax.set_ylabel("RUL Class")
    ax.set_title("Predicted vs True Class Over Cycle Life")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    ax = axes[1]
    for c in range(N_CLASSES):
        ax.plot(cycles, probs[:, c], color=CLASS_COLORS[c % len(CLASS_COLORS)],
                lw=1.5, label=names[c], alpha=0.85)
    ax.set_ylabel("Softmax Probability"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Predicted Class Probabilities")
    ax.legend(fontsize=8, ncol=N_CLASSES, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    error = pred_cls - true_cls
    ax.bar(cycles, error, width=1.0,
           color=["#E53935" if e else "#43A047" for e in error], alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Window End Cycle"); ax.set_ylabel("Class Error\n(pred - true)")
    ax.set_title("Classification Error per Window"); ax.grid(True, alpha=0.25)
    ax.legend(handles=[mpatches.Patch(color="#43A047", alpha=0.75, label="Correct"),
                       mpatches.Patch(color="#E53935", alpha=0.75, label="Incorrect")],
              fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_cells(all_results, out_path):
    short = [f"C{i}" for i in range(N_CLASSES)]
    n, ncols = len(all_results), 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("All Cells - Predicted vs True RUL Class",
                 fontsize=14, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    for ax, (cid, r) in zip(axes_flat, all_results.items()):
        cyc, p, t = r["end_cycles"], r["pred_classes"], r["true_classes"]
        ok = p == t
        ax.step(cyc, t, where="post", color="#1565C0", lw=1.8)
        ax.step(cyc, p, where="post", color="#E53935", lw=1.4, ls="--")
        ax.fill_between(cyc, -0.4, N_CLASSES - 0.6, where=ok, alpha=0.08,
                        color="green", step="post")
        ax.fill_between(cyc, -0.4, N_CLASSES - 0.6, where=~ok, alpha=0.14,
                        color="red", step="post")
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(short, fontsize=6)
        ax.set_ylim(-0.4, N_CLASSES - 0.6)
        ax.set_title(f"{cid}\nacc={r['accuracy']:.2%}", fontsize=7.5)
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2)

    for ax in axes_flat[len(all_results):]:
        ax.set_visible(False)

    fig.legend(handles=[
        Line2D([0], [0], color="#1565C0", lw=1.8, label="True class"),
        Line2D([0], [0], color="#E53935", lw=1.4, ls="--", label="Pred class"),
        mpatches.Patch(color="green", alpha=0.2, label="Correct"),
        mpatches.Patch(color="red", alpha=0.25, label="Incorrect")],
        loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(all_results, names, out_path):
    lives = [r["cycle_life"] for r in all_results.values()]
    vmin, vmax = min(lives), max(lives)
    cmap = colormaps.get_cmap("plasma")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("All Cells Overlay - Coloured by Cycle Life",
                 fontsize=13, fontweight="bold")

    for ax, key in zip(axes, ["true_classes", "pred_classes"]):
        for r in all_results.values():
            ax.step(r["end_cycles"], r[key], where="post",
                    color=cmap((r["cycle_life"] - vmin) / (vmax - vmin + 1e-9)),
                    alpha=0.6, lw=1.2)
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.4, N_CLASSES - 0.6)
        ax.set_title("Ground Truth" if key == "true_classes" else "Predicted",
                     fontsize=12)
        ax.set_xlabel("Window End Cycle"); ax.set_ylabel("RUL Class")
        ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02).set_label("Cycle Life")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_neol(eol_true, eol_pred, eol_err, mae, mape, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Near End-of-Life Estimation", fontsize=13, fontweight="bold")

    ax = axes[0]
    lim = [min(eol_true.min(), eol_pred.min()) - 50,
           max(eol_true.max(), eol_pred.max()) + 50]
    sc = ax.scatter(eol_true, eol_pred, c=np.abs(eol_err), cmap="RdYlGn_r",
                    s=60, edgecolors="k", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="|NEOL error| (cycles)")
    ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction")
    ax.fill_between(lim, [l - 100 for l in lim], [l + 100 for l in lim],
                    alpha=0.1, color="green", label="+/-100 cycle band")
    ax.set_xlabel("True NEOL (cycles)"); ax.set_ylabel("Predicted NEOL (cycles)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("NEOL Parity Plot"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(0.04, 0.92, f"MAE = {mae:.1f} cycles\nMAPE = {mape:.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax = axes[1]
    ax.hist(eol_err, bins=20, color="#1976D2", edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", lw=1.5, ls="--", label="Zero error")
    ax.axvline(eol_err.mean(), color="red", lw=1.5,
               label=f"Mean = {eol_err.mean():.1f} cycles")
    ax.set_xlabel("NEOL Prediction Error"); ax.set_ylabel("Count")
    ax.set_title("NEOL Error Distribution"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(true, pred, names, out_path):
    """Draw both matrix views and return the raw counts."""
    # labels= is required, not optional: without it sklearn infers the class
    # count from the data present and mislabels whenever a split happens to
    # contain fewer than N_CLASSES classes.
    cm = confusion_matrix(true, pred, labels=list(range(N_CLASSES)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, normalize, title in ((axes[0], True, "Row-normalised"),
                                 (axes[1], False, "Raw counts")):
        shown = (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
                 if normalize else cm.astype(float))
        im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max())
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11, fontweight="bold")

        thresh = shown.max() / 2
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                txt = f"{shown[i, j]:.1%}\n({cm[i, j]})" if normalize else f"{cm[i, j]}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if shown[i, j] > thresh else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


# ---- L5: reports -------------------------------------------------------

def build_reports(name, arch, cfg, ckpt_dir, data_dir, n_early, n_random,
                  all_results, cm, names, stats, figures):
    """Return (text_report, markdown_report)."""
    rows = []
    for cid in sorted(all_results):
        r = all_results[cid]
        t, p = neol_from_result(r)
        rows.append((cid, r["cycle_life"], t, p, p - t, r["accuracy"]))

    txt = []
    txt.append("=" * 72)
    txt.append(f"RUL classification report - {name}")
    txt.append("=" * 72)
    txt.append(f"Architecture   : {arch}")
    txt.append(f"Checkpoint dir : {ckpt_dir}")
    txt.append(f"Data dir       : {data_dir}")
    txt.append(f"Window         : n_early={n_early}  n_random={n_random}")
    txt.append(f"Loss           : {cfg.get('loss', '?')}")
    txt.append(f"Cells analysed : {len(all_results)}")
    txt.append("")
    txt.append("Classes:")
    for i, n in enumerate(names):
        txt.append(f"  {i}  {n}")
    txt.append("")
    txt.append("-" * 72)
    txt.append("Per-cell results")
    txt.append("-" * 72)
    txt.append(f'{"Cell":>14} | {"CycLife":>7} | {"NEOL_True":>9} | '
               f'{"NEOL_Pred":>9} | {"NEOL_Err":>8} | {"Acc":>7}')
    txt.append("-" * 72)
    for cid, life, t, p, err, acc in rows:
        txt.append(f'{cid:>14} | {life:7d} | {t:9d} | {p:9d} | {err:+8d} | {acc:7.2%}')
    txt.append("-" * 72)
    txt.append(f'{"MEAN":>14} | {"":>7} | {"":>9} | {"":>9} | '
               f'{stats["eol_mean_err"]:+8.1f} | {stats["acc_mean"]:7.2%}')
    txt.append(f'{"MAE":>14} | {"":>7} | {"":>9} | {"":>9} | {stats["eol_mae"]:8.1f} |')
    txt.append(f'{"MAPE":>14} | {"":>7} | {"":>9} | {"":>9} | {stats["eol_mape"]:7.1f}% |')
    txt.append("")
    txt.append("-" * 72)
    txt.append("Summary")
    txt.append("-" * 72)
    txt.append(f'Per-cell accuracy   mean={stats["acc_mean"]:.4f}  '
               f'std={stats["acc_std"]:.4f}  min={stats["acc_min"]:.4f}  '
               f'max={stats["acc_max"]:.4f}')
    txt.append(f'Overall window acc  {stats["overall_acc"]:.4f} '
               f'over {stats["n_windows"]:,} windows')
    txt.append(f'Exact class         {stats["exact"]:.2%}')
    txt.append(f'Within one class    {stats["within_one"]:.2%}')
    txt.append("")
    txt.append(f'NEOL MAE            {stats["eol_mae"]:.1f} cycles')
    txt.append(f'NEOL MAPE           {stats["eol_mape"]:.1f}%')
    txt.append(f'NEOL mean error     {stats["eol_mean_err"]:+.1f} cycles  '
               f'(+ = over-predict)')
    txt.append(f'NEOL std error      {stats["eol_std_err"]:.1f} cycles')
    txt.append("")
    txt.append("-" * 72)
    txt.append("Confusion matrix (rows = true, cols = predicted)")
    txt.append("-" * 72)
    header = "        " + "".join(f"{f'C{j}':>9}" for j in range(N_CLASSES))
    txt.append(header)
    for i in range(N_CLASSES):
        txt.append(f"  C{i}   " + "".join(f"{cm[i, j]:>9d}" for j in range(N_CLASSES)))

    md = []
    md.append(f"# RUL classification report — {name}")
    md.append("")
    md.append("| | |")
    md.append("|---|---|")
    md.append(f"| Architecture | {arch} |")
    md.append(f"| Checkpoint dir | `{ckpt_dir}` |")
    md.append(f"| Data dir | `{data_dir}` |")
    md.append(f"| Window | `n_early={n_early}`, `n_random={n_random}` |")
    md.append(f"| Loss | `{cfg.get('loss', '?')}` |")
    md.append(f"| Cells analysed | {len(all_results)} |")
    md.append(f"| Windows scored | {stats['n_windows']:,} |")
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Overall window accuracy | {stats['overall_acc']:.2%} |")
    md.append(f"| Per-cell accuracy (mean ± std) | {stats['acc_mean']:.2%} ± {stats['acc_std']:.2%} |")
    md.append(f"| Within one class | {stats['within_one']:.2%} |")
    md.append(f"| NEOL MAE | {stats['eol_mae']:.1f} cycles |")
    md.append(f"| NEOL MAPE | {stats['eol_mape']:.1f}% |")
    md.append(f"| NEOL mean error | {stats['eol_mean_err']:+.1f} cycles |")
    md.append("")
    md.append("NEOL is the first cycle predicted to fall in the most-degraded "
              "class, i.e. the estimate of when the cell is near the end of "
              "its life. A positive mean error means the model calls it late.")
    md.append("")
    md.append("## RUL classes")
    md.append("")
    md.append("| Class | Meaning |")
    md.append("|---|---|")
    for i, n in enumerate(names):
        md.append(f"| {i} | `{n}` |")
    md.append("")
    md.append("## Per-cell results")
    md.append("")
    md.append("| Cell | Cycle life | NEOL true | NEOL pred | NEOL error | Accuracy |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for cid, life, t, p, err, acc in rows:
        md.append(f"| `{cid}` | {life} | {t} | {p} | {err:+d} | {acc:.2%} |")
    md.append(f"| **MAE** | | | | **{stats['eol_mae']:.1f}** | **{stats['acc_mean']:.2%}** |")
    md.append(f"| **MAPE** | | | | **{stats['eol_mape']:.1f}%** | |")
    md.append("")
    md.append("## Confusion matrix")
    md.append("")
    md.append("Rows are the true class, columns the predicted one.")
    md.append("")
    md.append("| True \\ Pred | " + " | ".join(f"C{j}" for j in range(N_CLASSES)) + " |")
    md.append("|---|" + "---:|" * N_CLASSES)
    for i in range(N_CLASSES):
        md.append(f"| **C{i}** | " + " | ".join(str(cm[i, j]) for j in range(N_CLASSES)) + " |")
    md.append("")
    md.append("## Figures")
    md.append("")
    for caption, fname in figures:
        md.append(f"### {caption}")
        md.append("")
        md.append(f"![{caption}]({fname})")
        md.append("")

    return "\n".join(txt) + "\n", "\n".join(md) + "\n"


# ---- L6: entry point ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run sliding-window inference and write a report folder.")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Folder holding best_clf_bml.pt and model_config.json")
    parser.add_argument("--data_dir", default=None,
                        help="Folder of .npz features. Defaults to the "
                             "content_dir recorded in model_config.json.")
    parser.add_argument("--out_dir", default=None,
                        help="Defaults to ./predict_bml_<name>, where <name> "
                             "comes from the checkpoint folder.")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Cell ids to analyse. Default: every cell found.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    name = run_name(args.ckpt_dir)
    out_dir = args.out_dir or f"./predict_bml_{name}"

    model, cfg, arch = load_model(args.ckpt_dir, device)
    data_dir = resolve_data_dir(args, cfg)

    # Only train_clf_bml_V2 exposes --n_early/--n_random and records them.
    # The ES and transformer trainers always use the dataset module's
    # constants, so those checkpoints fall back to whatever they are now.
    n_early = cfg.get("n_early", N_EARLY)
    n_random = cfg.get("n_random", N_RANDOM)

    # All three trainers record n_input, so the fallback can be checked
    # rather than trusted: editing N_EARLY/N_RANDOM after training would
    # otherwise silently score the model on a window it never saw.
    recorded_input = cfg.get("n_input")
    if recorded_input is not None and n_early + n_random != recorded_input:
        sys.exit(
            f"ERROR: window mismatch. The checkpoint was trained with "
            f"n_input={recorded_input}, but this run would use "
            f"n_early={n_early} + n_random={n_random} = {n_early + n_random}.\n"
            f"       N_EARLY/N_RANDOM in dataset_clf_bml_v2.py have changed "
            f"since training. Restore them, or re-train."
        )

    print(f"Device      : {device}")
    print(f"Architecture: {arch}")
    print(f"Checkpoint  : {args.ckpt_dir}")
    print(f"Data        : {data_dir}")
    print(f"Window      : n_early={n_early}  n_random={n_random}")
    print(f"Output      : {out_dir}")

    for fname in ("dq_scaler_bml.pkl", "summary_scaler_bml.pkl"):
        if not os.path.isfile(os.path.join(args.ckpt_dir, fname)):
            sys.exit(f"ERROR: {fname} missing from {args.ckpt_dir}")
    scalers = (joblib.load(os.path.join(args.ckpt_dir, "dq_scaler_bml.pkl")),
               joblib.load(os.path.join(args.ckpt_dir, "summary_scaler_bml.pkl")))

    cells = load_all_npz(data_dir)
    if not cells:
        sys.exit(f"ERROR: no .npz files under {data_dir}")
    print(f"Loaded {len(cells)} cells")

    os.makedirs(out_dir, exist_ok=True)

    targets = sorted(cells) if args.cells is None else args.cells
    missing = [c for c in targets if c not in cells]
    if missing:
        print(f"WARNING: {len(missing)} cells not found: {missing[:5]}")
    targets = [c for c in targets if c in cells]

    names = class_names()
    all_results = {}
    for i, cid in enumerate(targets, 1):
        r = predict_cell(cells[cid], model, scalers, device, n_early, n_random)
        if len(r["end_cycles"]):
            all_results[cid] = r
        print(f"  [{i}/{len(targets)}] {cid}: acc={r['accuracy']:.4f} "
              f"({len(r['end_cycles'])} windows)")

    if not all_results:
        sys.exit("ERROR: every cell was too short for the window - nothing to report.")

    eol_true, eol_pred = [], []
    for r in all_results.values():
        t, p = neol_from_result(r)
        eol_true.append(t); eol_pred.append(p)
    eol_true = np.array(eol_true)
    eol_pred = np.array(eol_pred)
    eol_err = eol_pred - eol_true

    accs = np.array([r["accuracy"] for r in all_results.values()])
    all_true = np.concatenate([r["true_classes"] for r in all_results.values()])
    all_pred = np.concatenate([r["pred_classes"] for r in all_results.values()])
    off = np.abs(all_true - all_pred)

    stats = {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
        "acc_min": float(accs.min()), "acc_max": float(accs.max()),
        "overall_acc": float((all_true == all_pred).mean()),
        "n_windows": int(all_true.size),
        "exact": float((off == 0).mean()),
        "within_one": float((off <= 1).mean()),
        "eol_mae": float(np.mean(np.abs(eol_err))),
        "eol_mape": float(np.mean(np.abs(eol_err) / (eol_true + 1e-6)) * 100),
        "eol_mean_err": float(eol_err.mean()),
        "eol_std_err": float(eol_err.std()),
    }

    first_cid = sorted(all_results)[0]
    single_png = f"{first_cid}_clf_prediction.png"
    plot_single_cell(first_cid, all_results[first_cid], names,
                     os.path.join(out_dir, single_png))
    plot_all_cells(all_results, os.path.join(out_dir, "all_cells_clf.png"))
    plot_overlay(all_results, names, os.path.join(out_dir, "overlay_all_cells_clf.png"))
    plot_neol(eol_true, eol_pred, eol_err, stats["eol_mae"], stats["eol_mape"],
              os.path.join(out_dir, "eol_parity_clf.png"))
    cm = plot_confusion(all_true, all_pred, names,
                        os.path.join(out_dir, "confusion_matrix.png"))

    figures = [
        (f"Single cell — {first_cid}", single_png),
        ("All cells", "all_cells_clf.png"),
        ("Overlay by cycle life", "overlay_all_cells_clf.png"),
        ("NEOL parity and error", "eol_parity_clf.png"),
        ("Confusion matrix", "confusion_matrix.png"),
    ]

    txt, md = build_reports(name, arch, cfg, args.ckpt_dir, data_dir,
                            n_early, n_random, all_results, cm, names,
                            stats, figures)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(out_dir, "predict.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print()
    print(txt)
    print(f"Wrote 5 figures, report.txt and predict.md to {out_dir}")


if __name__ == "__main__":
    main()
