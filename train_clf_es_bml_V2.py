"""train_clf_es_bml_V2.py - Sparse CMA-ES trainer for the GRU weights of a
CNN+GRU RUL classifier on BatteryML features.

CNN, summary_proj, post_gru_drop and head stay frozen at their pretrained
or randomly-initialised values; only a keep_ratio fraction of the largest
GRU weights is evolved by CMA-ES. No gradient descent runs anywhere in this
file.

How to run:
    python train_clf_es_bml_V2.py --content_dir ./content_bml --output_dir ./checkpoints_es_bml

Fine-tune GRU weights on top of an existing gradient-trained checkpoint:
    python train_clf_es_bml_V2.py --content_dir ./content_bml/HUST \
        --output_dir ./checkpoints_clf_bml_HUST_es \
        --pretrain_ckpt ./checkpoints_clf_bml_HUST/best_clf_bml.pt

Main outputs, all under --output_dir:
    best_clf_bml.pt          Weights only, so the notebook can load_state_dict.
    model_config.json        The architecture those weights belong to.
    dq_scaler_bml.pkl        Scalers -- from the pretrain checkpoint's
    summary_scaler_bml.pkl   directory if --pretrain_ckpt is set, else fresh.
    clf_pred_bml.npy         Test-set predictions and ground truth.
    clf_true_bml.npy
    weight_heatmaps/gru_weight_heatmaps.png   Before/after ES, unless
                                               --no_weight_heatmaps.
    Info_log.txt             Mirror of everything printed to console.
"""

import argparse
import copy
import json
import os
import random
import sys
import traceback
from datetime import datetime

import cma
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

from dataset_clf_bml_v2 import (N_CLASSES, N_INPUT, RUL_EDGES, V_BINS,
                                build_clf_dataloaders)
from model_clf import BatteryRULClassifier, OrdinalLoss, evaluate, predict_cls
from train_utils import Tee, class_names, print_model_summary, set_seed


def save_model_config(path: str, args, summary_feats: int) -> None:
    config = {
        "cnn_dim": args.cnn_dim,
        "gru_dim": args.gru_dim,
        "gru_layers": args.gru_layers,
        "dropout": 0.0,
        "summary_feats": summary_feats,
        "n_classes": N_CLASSES,
        "n_input": N_INPUT,
        "v_bins": V_BINS,
        "loss": args.loss,
        "seed": args.seed,
        "content_dir": args.content_dir,
        "es_keep_ratio": args.es_keep_ratio,
        "es_keep_bias": args.es_keep_bias,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---- sparse ES mask --------------------------------------------------------

_ES_MODULES = ("gru",)


def build_sparse_es_mask(model: nn.Module, keep_ratio: float = 0.3,
                         keep_bias: bool = True,
                         zero_inactive: bool = False) -> list:
    keep_ratio = float(keep_ratio)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")

    meta = []
    for name, p in model.named_parameters():
        if not any(name.startswith(m) for m in _ES_MODULES):
            continue
        arr = p.data.detach().cpu().numpy()
        if arr.ndim == 0:
            continue

        if p.ndim == 1 and keep_bias:
            mask = np.ones(arr.shape, dtype=bool)
        else:
            flat_abs = np.abs(arr).ravel()
            k = max(1, int(np.ceil(flat_abs.size * keep_ratio)))
            if k >= flat_abs.size:
                mask = np.ones(arr.shape, dtype=bool)
            else:
                threshold = np.partition(flat_abs, -k)[-k]
                mask = np.abs(arr) >= threshold   # keep the LARGEST weights

        if zero_inactive:
            pruned = arr.copy()
            pruned[~mask] = 0.0
            p.data.copy_(torch.tensor(pruned, dtype=p.dtype, device=p.device))

        meta.append({"name": name, "param": p, "shape": tuple(p.shape),
                     "mask": mask, "active": int(mask.sum()),
                     "total": int(mask.size)})

    n_sparse = sum(m["active"] for m in meta)
    n_total = sum(m["total"] for m in meta)
    n_model = sum(p.numel() for p in model.parameters())
    if n_sparse == 0:
        raise RuntimeError("Sparse ES mask is empty. Check keep_ratio.")

    print(f"\nSparse ES mask (only {_ES_MODULES} evolves; everything else frozen):")
    print("=" * 80)
    print(f"  {'Parameter':<45} {'Active/Total':>20} {'Ratio':>10}")
    print("-" * 80)
    for m in meta:
        ratio = 100.0 * m["active"] / m["total"]
        print(f"  {m['name']:<45} {m['active']:>8,}/{m['total']:<8,} {ratio:>9.2f}%")
    print("-" * 80)
    print(f"  {'ES active / model total':<45} {n_sparse:>8,}/{n_model:<8,} "
          f"{100.0*n_sparse/n_model:>9.2f}%")
    print("=" * 80 + "\n")
    return meta


def get_flat_params(meta: list) -> np.ndarray:
    """Read the active elements of every masked tensor into one vector."""
    return np.concatenate([
        m["param"].data.detach().cpu().numpy()[m["mask"]].ravel()
        for m in meta
    ]).astype(np.float64)


def set_flat_params(meta: list, flat: np.ndarray) -> None:
    """Write a flat vector back into the active elements it came from.

    Raises:
        ValueError: If flat has more values than the mask has active slots.
    """
    offset = 0
    flat = np.asarray(flat)
    for m in meta:
        p, mask = m["param"], m["mask"]
        size = int(mask.sum())
        arr = p.data.detach().cpu().numpy().copy()
        arr[mask] = flat[offset:offset + size]
        p.data.copy_(torch.tensor(arr, dtype=p.dtype, device=p.device))
        offset += size
    if offset != len(flat):
        raise ValueError(f"Unused values: used {offset}, got {len(flat)}")


# ---- fitness ---------------------------------------------------------------


@torch.no_grad()


def accuracy(model: nn.Module, loader, device: torch.device) -> float:
    """Fraction of correct predictions over one pass of a loader."""
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch["dq"].to(device), batch["summary"].to(device))
        labels = batch["label"].to(device)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total if total else 0.0


def _make_subset_loader(loader: DataLoader, max_batches: int = 10) -> DataLoader:
    """A fresh random subset of a loader's dataset, for a fast fitness signal."""
    dataset = loader.dataset
    n = min(max_batches * loader.batch_size, len(dataset))
    idx = np.random.choice(len(dataset), size=n, replace=False)
    return DataLoader(Subset(dataset, idx), batch_size=loader.batch_size,
                      shuffle=False, num_workers=0, pin_memory=loader.pin_memory)


def _worker_fitness(rank: int, flat_params: np.ndarray, meta_list: list,
                    model_state: dict, model_kwargs: dict, batches: list,
                    result_queue: mp.Queue) -> None:
    """Evaluate one CMA-ES offspring's accuracy in a subprocess.

    model_kwargs is the caller's own args (cnn_dim, gru_dim, ...), passed
    through as-is rather than guessed back from a layer's tensor shape --
    the old code inferred cnn_dim as head[0].in_features // 2, which
    actually equals gru_dim (verified: cnn_dim=64, gru_dim=32 reconstructs
    to 32), silently correct only when the two happened to be equal, and a
    guaranteed shape-mismatch crash in load_state_dict otherwise.

    On failure this prints the full traceback before reporting fitness 0.0,
    so a worker crash reads as a crash, not as a genuinely bad offspring.

    Args:
        rank: This offspring's index, to route the result back correctly.
        flat_params: This offspring's parameter vector (active slots only).
        meta_list: Per-tensor {name, mask} pairs (no param/tensor objects --
            those are not picklable across a process boundary).
        model_state: The base model's full state_dict, CPU tensors.
        model_kwargs: Constructor arguments for BatteryRULClassifier.
        batches: Pre-fetched (dq, summary, labels) CPU tensors to score on.
        result_queue: Where (rank, accuracy) is sent back.
    """
    try:
        device = torch.device("cpu")
        model = BatteryRULClassifier(**model_kwargs).to(device)
        model.load_state_dict(model_state)
        model.eval()

        offset = 0
        flat = np.asarray(flat_params)
        params = dict(model.named_parameters())
        for meta in meta_list:
            mask, name = meta["mask"], meta["name"]
            size = int(mask.sum())
            p = params[name]
            arr = p.data.detach().cpu().numpy().copy()
            arr[mask] = flat[offset:offset + size]
            p.data.copy_(torch.tensor(arr, dtype=p.dtype, device=device))
            offset += size

        correct = total = 0
        with torch.no_grad():
            for dq, summary, labels in batches:
                logits = model(dq.to(device), summary.to(device))
                correct += (logits.argmax(dim=1) == labels.to(device)).sum().item()
                total += len(labels)
        result_queue.put((rank, correct / total if total else 0.0))
    except Exception:
        print(f"[worker {rank}] FAILED:\n{traceback.format_exc()}")
        result_queue.put((rank, 0.0))


def fitness_parallel(model: nn.Module, solutions: list, loader,
                     model_kwargs: dict, meta: list,
                     n_workers: int = 4, l1_lambda: float = 0.0) -> list:
    """Evaluate every offspring in one CMA-ES generation in parallel.

    Batches are pre-fetched to CPU once and shared read-only across workers;
    each worker rebuilds its own model copy on CPU (avoids GPU contention)
    and applies one offspring's parameters before scoring.

    Args:
        model: The base model (for its current state_dict).
        solutions: This generation's parameter vectors from CMA-ES.
        loader: Batches to score against.
        model_kwargs: Constructor arguments for BatteryRULClassifier,
            supplied by the caller rather than reconstructed from tensor
            shapes.
        meta: Sparse-mask metadata from build_sparse_es_mask.
        n_workers: Number of subprocesses.
        l1_lambda: Weight of an L1 penalty on the active parameters, added
            to the loss (so it *lowers* fitness for large weights).

    Returns:
        Negated-accuracy-plus-penalty for each solution, in input order --
        this is what CMA-ES minimises.
    """
    batches = [(b["dq"].cpu(), b["summary"].cpu(), b["label"].cpu())
              for b in loader]
    meta_list = [{"name": m["name"], "mask": m["mask"]} for m in meta]
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    result_queue = mp.Queue()
    n_sol = len(solutions)
    chunk_size = max(1, (n_sol + n_workers - 1) // n_workers)
    processes = []

    for w in range(n_workers):
        start, end = w * chunk_size, min((w + 1) * chunk_size, n_sol)
        if start >= n_sol:
            break
        for i, sol in enumerate(solutions[start:end]):
            proc = mp.Process(target=_worker_fitness,
                              args=(start + i, sol.astype(np.float32), meta_list,
                                    model_state, model_kwargs, batches, result_queue),
                              daemon=True)
            proc.start()
            processes.append(proc)

    results = {}
    for _ in processes:
        rank, acc = result_queue.get()
        results[rank] = acc
    for proc in processes:
        proc.join(timeout=60)

    fitnesses = []
    for i in range(n_sol):
        acc = results.get(i, 0.0)
        penalty = l1_lambda * float(np.abs(solutions[i])).mean() if l1_lambda > 0 else 0.0
        fitnesses.append(-acc + penalty)
    return fitnesses

# ---- CMA-ES loop -----------------------------------------------------------


def cmaes_run(model: nn.Module, train_loader, val_loader, test_loader,
              meta: list, model_kwargs: dict, device: torch.device,
              n_gen: int = 200, sigma: float = 0.01, popsize: int = None,
              output_dir: str = ".", seed: int = 42, acc_gap_tol: float = 0.01,
              acc_min: float = 0.70, l1_lambda: float = 0.0,
              n_workers: int = 4) -> float:
    """Evolve the masked GRU weights with CMA-ES. No gradient descent runs.

    Two different accuracy measurements happen per generation, and they are
    deliberately not the same thing:

    * Selection uses ``fast_loader`` -- a fresh random 10-batch subset of the
      validation split, redrawn every generation. Cheap enough to afford
      popsize evaluations per generation, but noisy.
    * The printed train/val/test columns re-evaluate the generation's best
      offspring on the full loaders. Expensive, so they inform the human
      reading the log rather than the search itself.

    Args:
        model: The model to evolve. Left holding the best parameters found.
        train_loader: Monitoring only -- never drives selection.
        val_loader: Source of the fitness subsets, and of the returned score.
        test_loader: Monitoring only.
        meta: Sparse-mask metadata from build_sparse_es_mask.
        model_kwargs: Constructor arguments, forwarded to the parallel workers.
        device: Device the monitoring passes run on.
        n_gen: Maximum generations.
        sigma: Initial step size.
        popsize: Population size, or None for the CMA default.
        output_dir: Where best_clf_bml.pt is written on every improvement.
        seed: Seed for the strategy.
        acc_gap_tol: Stop once train and val accuracy are this close.
        acc_min: ...but only once val accuracy has reached at least this.
        l1_lambda: Weight of an L1 penalty on the evolved parameters.
        n_workers: Subprocesses for parallel fitness; 1 runs sequentially.

    Returns:
        Accuracy of the best parameters on the FULL validation loader.
    """
    x0 = get_flat_params(meta).astype(np.float64)
    n_model = sum(p.numel() for p in model.parameters())

    train_acc = accuracy(model, train_loader, device)
    val_acc = accuracy(model, val_loader, device)
    test_acc = accuracy(model, test_loader, device)
    print(f"GRU ES params: {len(x0):,} active / {n_model:,} model total")
    print(f"Parallel workers: {n_workers}")
    print(f"Initial  train={train_acc:.4f}  val={val_acc:.4f}  "
          f"test={test_acc:.4f}  sigma={sigma:.4f}")

    options = {"seed": seed, "maxiter": n_gen, "verbose": -9,
               "tolx": 1e-8, "tolfun": 1e-7, "CMA_diagonal": True}
    if popsize is not None:
        options["popsize"] = popsize

    strategy = cma.CMAEvolutionStrategy(x0, sigma, options)
    best_params = x0.copy()
    # One tracker, updated every generation that improves. The old file kept
    # two (best_acc, best_acc_x): only the second was ever updated, while the
    # first stayed frozen at its pre-loop value and yet drove both the
    # early-stop gate and the return value.
    best_fitness = 0.0

    header = (f"{'Gen':>5} | {'TrainAcc':>8} | {'ValAcc':>8} | "
              f"{'TestAcc':>8} | {'Sigma':>10} | {'Improved':>9}")
    print("\n" + header)
    print("-" * len(header))

    while not strategy.stop():
        fast_loader = _make_subset_loader(val_loader, max_batches=10)
        solutions = strategy.ask()

        if n_workers > 1:
            fitnesses = fitness_parallel(
                model=model, solutions=solutions, loader=fast_loader,
                model_kwargs=model_kwargs, meta=meta,
                n_workers=n_workers, l1_lambda=l1_lambda)
        else:
            fitnesses = []
            for sol in solutions:
                set_flat_params(meta, sol.astype(np.float32))
                acc = accuracy(model, fast_loader, device)
                penalty = (l1_lambda * float(np.abs(sol).mean())
                           if l1_lambda > 0 else 0.0)
                fitnesses.append(-acc + penalty)

        strategy.tell(solutions, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        gen_best = -fitnesses[best_idx]
        improved = gen_best > best_fitness
        if improved:
            best_fitness = gen_best
            best_params = solutions[best_idx].copy()
            set_flat_params(meta, best_params.astype(np.float32))
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_clf_bml.pt"))

        set_flat_params(meta, solutions[best_idx].astype(np.float32))
        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        set_flat_params(meta, best_params.astype(np.float32))

        print(f"{strategy.countiter:5d} | {train_acc:8.4f} | {val_acc:8.4f} | "
              f"{test_acc:8.4f} | {strategy.sigma:10.6f} | "
              f"{'yes' if improved else 'no':>9}")

        if val_acc >= acc_min and abs(val_acc - train_acc) <= acc_gap_tol:
            print(f"\nEarly stop: val={val_acc:.4f} train={train_acc:.4f} "
                  f"gap={abs(val_acc - train_acc):.4f} <= tol={acc_gap_tol}")
            break

    print(f"\nCMA-ES stopped: {strategy.stop()}")
    set_flat_params(meta, best_params.astype(np.float32))
    # best_fitness tracked progress against the noisy fast_loader subsets.
    # Recompute once on the full validation loader so the number handed back
    # is comparable to anything else that says "validation accuracy".
    return accuracy(model, val_loader, device)


# ---- weight heatmaps -------------------------------------------------------


@torch.no_grad()


def save_weight_heatmaps(model_before: nn.Module, model_after: nn.Module,
                         output_dir: str, max_cols: int = 256,
                         include_bias: bool = False) -> None:
    """Plot every evolved tensor before and after ES, side by side.

    Only tensors CMA-ES was allowed to touch are plotted. With
    _ES_MODULES = ("gru",) that is genuinely the GRU alone -- the old file's
    tuple matched every submodule, so this "ES modules only" filter plotted
    the whole model despite the filename.

    Args:
        model_before: Snapshot taken before ES ran.
        model_after: The evolved model.
        output_dir: Directory for gru_weight_heatmaps.png.
        max_cols: Column cap per tensor, so wide matrices stay legible.
        include_bias: Also plot 1-D bias tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    before_dict = dict(model_before.named_parameters())
    after_dict = dict(model_after.named_parameters())

    valid_layers = [
        name for name in before_dict
        if name in after_dict
        and before_dict[name].detach().cpu().numpy().ndim > 0
        and any(name.startswith(m) for m in _ES_MODULES)
        and (include_bias or before_dict[name].ndim > 1)
    ]
    if not valid_layers:
        print("No layers to plot in heatmap.")
        return

    n_layers = len(valid_layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, max(3 * n_layers, 8)))
    if n_layers == 1:
        axes = np.array([axes])

    for row, name in enumerate(valid_layers):
        w_b = before_dict[name].detach().cpu().numpy()
        w_a = after_dict[name].detach().cpu().numpy()
        w_b2 = (w_b.reshape(1, -1) if w_b.ndim == 1
                else w_b.reshape(w_b.shape[0], -1))[:, :max_cols]
        w_a2 = (w_a.reshape(1, -1) if w_a.ndim == 1
                else w_a.reshape(w_a.shape[0], -1))[:, :max_cols]
        vmax = max(np.abs(w_b2).max(), np.abs(w_a2).max())
        axes[row, 0].imshow(w_b2, aspect="auto", cmap="seismic",
                            vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"Before ES\n{name}")
        axes[row, 0].set_ylabel(str(w_b2.shape))
        im = axes[row, 1].imshow(w_a2, aspect="auto", cmap="seismic",
                                 vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f"After ES\n{name}")

    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.995,
                 pad=0.01).set_label("Weight Value")
    save_path = os.path.join(output_dir, "gru_weight_heatmaps.png")
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {save_path}")


# ---- training --------------------------------------------------------------


def train(args) -> None:
    """Run one sparse-CMA-ES job end to end and write every artefact."""
    set_seed(args.seed)
    mp.set_start_method("spawn", force=True)   # required for CUDA + workers

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "Info_log.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Logging to: {log_path}")

    print("\nLoading BML data...")
    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir=args.content_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.pretrain_ckpt:
        ckpt_dir = os.path.dirname(args.pretrain_ckpt)
        dq_scaler = joblib.load(os.path.join(ckpt_dir, "dq_scaler_bml.pkl"))
        summary_scaler = joblib.load(
            os.path.join(ckpt_dir, "summary_scaler_bml.pkl"))
    else:
        dq_scaler, summary_scaler = scalers

    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler,
                os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    # dropout stays 0.0: a stochastic forward pass would make one offspring
    # score differently each time it is measured, which breaks CMA-ES's
    # assumption that fitness is a deterministic function of the parameters.
    model_kwargs = {
        "cnn_dim": args.cnn_dim, "gru_dim": args.gru_dim,
        "gru_layers": args.gru_layers, "summary_feats": summary_feats,
        "n_classes": N_CLASSES, "dropout": 0.0,
    }
    model = BatteryRULClassifier(**model_kwargs).to(device)
    print_model_summary(model, N_INPUT, V_BINS, summary_feats, device)
    save_model_config(os.path.join(args.output_dir, "model_config.json"),
                      args, summary_feats)

    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location=device,
                          weights_only=True)
        model_state = model.state_dict()
        matched, skipped = {}, []
        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                shape = (list(model_state[k].shape) if k in model_state
                         else "missing")
                skipped.append(f"{k}: ckpt{list(v.shape)} vs {shape}")
        model.load_state_dict(matched, strict=False)
        print(f"Pretrained weights loaded from: {args.pretrain_ckpt}")
        print(f"  Matched: {len(matched)}/{len(ckpt)} keys")
        for s in skipped:
            print(f"    {s}")
    else:
        print("No pretrained weights - starting from random init.")

    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn = predict_cls
    else:
        criterion = nn.CrossEntropyLoss()

        def pred_fn(logits):
            return logits.argmax(dim=1)
    print(f"Loss: {type(criterion).__name__}\n")

    meta = build_sparse_es_mask(model, keep_ratio=args.es_keep_ratio,
                                keep_bias=args.es_keep_bias,
                                zero_inactive=args.es_zero_inactive)
    model_before_es = copy.deepcopy(model).cpu()

    best_val_acc = cmaes_run(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, meta=meta, model_kwargs=model_kwargs,
        device=device, n_gen=args.n_gen, sigma=args.sigma,
        popsize=args.popsize, output_dir=args.output_dir, seed=args.seed,
        acc_gap_tol=args.acc_gap_tol, acc_min=args.acc_min,
        l1_lambda=args.es_l1_lambda, n_workers=args.n_workers,
    )
    print(f"Best val accuracy: {best_val_acc:.4f}")

    print("\n" + "=" * 60)
    best_path = os.path.join(args.output_dir, "best_clf_bml.pt")
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion,
                                           pred_fn, device)
    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, pred, labels=list(range(N_CLASSES)),
                                target_names=class_names(RUL_EDGES), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred_bml.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true_bml.npy"), true)
    print(f"\nSaved to {args.output_dir}/")

    if not args.no_weight_heatmaps:
        save_weight_heatmaps(
            model_before=model_before_es,
            model_after=copy.deepcopy(model).cpu(),
            output_dir=os.path.join(args.output_dir, "weight_heatmaps"),
            max_cols=args.heatmap_max_cols,
            include_bias=args.heatmap_include_bias)

    sys.stdout = sys.__stdout__
    log_f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse CMA-ES trainer for the GRU weights of the "
                    "CNN+GRU RUL classifier.")
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--output_dir", default="./checkpoints_es_bml")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn_dim", type=int, default=32)
    parser.add_argument("--gru_dim", type=int, default=32)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--pretrain_ckpt", default=None,
                        help="Gradient-trained checkpoint to evolve on top of.")
    parser.add_argument("--loss", default="ordinal",
                        choices=["cross_entropy", "ordinal"])
    # ---- CMA-ES
    parser.add_argument("--n_gen", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--acc_gap_tol", type=float, default=0.01)
    parser.add_argument("--acc_min", type=float, default=0.70)
    # ---- sparse ES
    parser.add_argument("--es_keep_ratio", type=float, default=0.3)
    parser.add_argument("--es_l1_lambda", type=float, default=0.0)
    parser.add_argument("--es_keep_bias", action="store_true", default=True)
    parser.add_argument("--es_zero_inactive", action="store_true")
    # ---- parallelism
    parser.add_argument("--n_workers", type=int, default=4,
                        help="CPU workers for offspring fitness; 1 = sequential")
    # ---- heatmaps
    parser.add_argument("--no_weight_heatmaps", action="store_true")
    parser.add_argument("--heatmap_max_cols", type=int, default=256)
    parser.add_argument("--heatmap_include_bias", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()