"""
train_clf_bml.py - Train CNN+GRU classifier on BatteryML-derived features.

How to run:
    python train_clf_bml.py --content_dir ./content_bml --output_dir ./checkpoints_clf_bml

Train one BML family/subfolder only:
    python train_clf_bml.py --content_dir ./content_bml/MATR --output_dir ./checkpoints_clf_bml_MATR --es_mode 0

Main outputs:
    checkpoints_clf_bml/best_clf_bml.pt
    checkpoints_clf_bml/dq_scaler_bml.pkl
    checkpoints_clf_bml/summary_scaler_bml.pkl
    checkpoints_clf_bml/clf_pred_bml.npy
    checkpoints_clf_bml/clf_true_bml.npy
"""

import argparse
import json
import os
import random
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from dataset_clf_bml_v2 import (N_CLASSES, N_EARLY, N_RANDOM, RUL_EDGES,
                                V_BINS, build_clf_dataloaders)
from model_clf import (BatteryRULClassifier, OrdinalLoss, evaluate,
                       predict_cls, train_epoch)
from train_utils import Tee, class_names, print_model_summary, set_seed


def save_model_config(path: str, args, summary_feats: int) -> None:
    config = {
        "cnn_dim": args.cnn_dim,
        "gru_dim": args.gru_dim,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
        "summary_feats": summary_feats,
        "n_classes": N_CLASSES,
        "n_early": args.n_early,
        "n_random": args.n_random,
        "n_input": args.n_early + args.n_random,
        "v_bins": V_BINS,
        "loss": args.loss,
        "seed": args.seed,
        "content_dir": args.content_dir,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# ---- fitness -------------------------------------------------------------

@torch.no_grad()


def accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch["dq"].to(device), batch["summary"].to(device))
        labels = batch["label"].to(device)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return correct / total

# -------------------------------------------------------------------------
# CMA-ES run — called when gradient training stagnates
# -------------------------------------------------------------------------

# ---- CMA-ES fine-tuning --------------------------------------------------


def cmaes_run(model: nn.Module, val_loader, device: torch.device,
              n_gen: int = 50, sigma: float = 0.02, popsize: int = None,
              seed: int = 42) -> float:
    try:
        import cma
    except ImportError as exc:                   # optional dependency
        raise ImportError(
            "CMA-ES modes need the 'cma' package: pip install cma") from exc

    x0 = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    x0 = x0.astype(np.float64)
    best_acc = accuracy(model, val_loader, device)
    print(f"  CMA-ES start - val: {best_acc:.4f}  n_params: {len(x0):,}  "
          f"sigma: {sigma:.4f}")

    options = {"seed": seed, "maxiter": n_gen, "verbose": -9,
               "tolx": 1e-8, "tolfun": 1e-7, "CMA_diagonal": True}
    if popsize is not None:
        options["popsize"] = popsize

    strategy = cma.CMAEvolutionStrategy(x0, sigma, options)
    best_params = x0.copy()

    header = f"  {'Gen':>5} | {'Sigma':>10} | {'GenBest':>8} | {'ValAcc':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    while not strategy.stop():
        solutions = strategy.ask()
        losses = []
        for solution in solutions:
            vector_to_parameters(
                torch.as_tensor(solution, dtype=torch.float32, device=device),
                model.parameters())
            # vector_to_parameters overwrites the GRU's weight tensors one
            # at a time, breaking the single contiguous buffer cuDNN wants.
            # Without this, cuDNN re-flattens on every forward call inside
            # accuracy() instead of once here -- measured at 27,563 warnings
            # and no generation finishing in over 5 minutes on a 12-cell
            # smoke set before this fix.
            model.gru.flatten_parameters()
            losses.append(-accuracy(model, val_loader, device))  # CMA minimises
        strategy.tell(solutions, losses)

        best_idx = int(np.argmin(losses))
        gen_best = -losses[best_idx]
        if gen_best > best_acc:
            best_acc = gen_best
            best_params = solutions[best_idx].copy()
        print(f"  {strategy.countiter:5d} | {strategy.sigma:10.6f} | "
              f"{gen_best:8.4f} | {best_acc:8.4f}")

    print(f"  CMA-ES stopped: {strategy.stop()}")
    vector_to_parameters(
        torch.as_tensor(best_params, dtype=torch.float32, device=device),
        model.parameters())
    model.gru.flatten_parameters()
    return best_acc

# ---- training loop -------------------------------------------------------


def train(args) -> None:
    """Run one training job end to end and write every artefact."""
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "Info_log.txt")
    log_f    = open(log_path, "w", encoding="utf-8")
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
        n_early=args.n_early,
        n_random=args.n_random,
    )

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler,
                os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifier(
        cnn_dim=args.cnn_dim, gru_dim=args.gru_dim, gru_layers=args.gru_layers,
        summary_feats=summary_feats, n_classes=N_CLASSES, dropout=args.dropout,
    ).to(device)
    print_model_summary(model, args.n_early + args.n_random, V_BINS,
                        summary_feats, device)
    save_model_config(os.path.join(args.output_dir, "model_config.json"),
                      args, summary_feats)

    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn = predict_cls
    else:
        criterion = nn.CrossEntropyLoss()

        def pred_fn(logits):
            return logits.argmax(dim=1)
    print(f"Loss: {type(criterion).__name__}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=args.lr * 0.01)

    best_path = os.path.join(args.output_dir, "best_clf_bml.pt")
    best_val_acc = -1.0
    stagnant = 0

    if args.es_mode == 3:
        print("ES mode 3: CMA-ES only (no gradient training)")
        best_val_acc = cmaes_run(model, val_loader, device, n_gen=50,
                                 sigma=args.sigma, seed=args.seed)
        torch.save(model.state_dict(), best_path)
        print(f"  CMA-ES best val acc: {best_val_acc:.4f} - checkpoint saved.")
    else:
        header = (f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | "
                  f"{'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}")
        print(header)
        print("-" * len(header))
        prev_lr = optimizer.param_groups[0]["lr"]

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                          pred_fn, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion,
                                             pred_fn, device)
            scheduler.step(va_acc)

            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
                  f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")
            if lr != prev_lr:
                print(f"        learning rate {prev_lr:.2e} -> {lr:.2e}")
                prev_lr = lr

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                stagnant = 0
                torch.save(model.state_dict(), best_path)
            else:
                stagnant += 1

            if args.es_mode == 2 and stagnant >= args.es_patience:
                print(f"\n[Epoch {epoch}] No improvement for {stagnant} "
                      f"epochs - running CMA-ES...")
                cma_acc = cmaes_run(model, val_loader, device, n_gen=10,
                                    sigma=args.sigma, seed=args.seed)
                if cma_acc > best_val_acc:
                    best_val_acc = cma_acc
                    torch.save(model.state_dict(), best_path)
                    print(f"  CMA-ES improved val acc to {best_val_acc:.4f} "
                          f"- checkpoint saved.")
                else:
                    # Gradient training must not continue from parameters
                    # that scored worse than the ones already on disk.
                    model.load_state_dict(torch.load(
                        best_path, map_location=device, weights_only=True))
                    print(f"  CMA-ES did not improve ({cma_acc:.4f} <= "
                          f"{best_val_acc:.4f}) - restored gradient best.")
                stagnant = 0

        if args.es_mode == 1:
            print("\nES mode 1: running CMA-ES after gradient training...")
            model.load_state_dict(torch.load(best_path, map_location=device,
                                             weights_only=True))
            cma_acc = cmaes_run(model, val_loader, device, n_gen=50,
                                sigma=args.sigma, seed=args.seed)
            if cma_acc > best_val_acc:
                best_val_acc = cma_acc
                torch.save(model.state_dict(), best_path)
                print(f"  CMA-ES improved val acc to {best_val_acc:.4f} "
                      f"- checkpoint saved.")
            else:
                model.load_state_dict(torch.load(best_path, map_location=device,
                                                 weights_only=True))
                print(f"  CMA-ES did not improve ({cma_acc:.4f} <= "
                      f"{best_val_acc:.4f}) - restored gradient best.")

    print("\n" + "=" * 60)
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

    sys.stdout = sys.__stdout__
    log_f.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CNN+GRU RUL classifier on BatteryML features.")
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--output_dir", default="./checkpoints_clf_bml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn_dim", type=int, default=32)
    parser.add_argument("--gru_dim", type=int, default=32)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_early", type=int, default=N_EARLY,
                        help="Leading cycles carried by every sample.")
    parser.add_argument("--n_random", type=int, default=N_RANDOM,
                        help="Consecutive cycles drawn from past the lead-in.")
    parser.add_argument("--loss", default="ordinal",
                        choices=["cross_entropy", "ordinal"])
    parser.add_argument("--sigma", type=float, default=0.02,
                        help="Initial CMA-ES step size.")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Stagnant epochs before es_mode 2 fires CMA-ES.")
    parser.add_argument("--es_mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="0=gradient only, 1=CMA-ES after gradient, "
                             "2=CMA-ES on stagnation, 3=CMA-ES only")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()