"""train_clf_bml_transformer_V2.py - Train a CNN+Transformer RUL classifier
on BatteryML features.

Self-contained: the model, loss and train/eval loops live here rather than
being imported from train_clf_transformer.py, so this file pulls in no other
trainer and no other dataset's pipeline.

Unlike the file it replaces, the positional encoding is actually applied.
The old model built a sinusoidal table in _build_pos_enc and never called
it, leaving self-attention blind to cycle order -- measured on that
architecture, f(x) and f(reversed x) matched to 0.000000, meaning a window
of 16 cycles was processed as an unordered bag. For a task about
degradation over time that discards the signal that matters most.

How to run:
    python train_clf_bml_transformer_V2.py --content_dir ./content_bml
        --output_dir ./checkpoints_clf_bml_tf

Train one BML family/subfolder only:
    python train_clf_bml_transformer_V2.py --content_dir ./content_bml/MATR
        --output_dir ./checkpoints_clf_bml_MATR_tf

Main outputs, all under --output_dir:
    best_clf_bml.pt          Weights only, so the notebook can load_state_dict.
    model_config.json        The architecture those weights belong to.
    dq_scaler_bml.pkl        Scalers fitted on the training split.
    summary_scaler_bml.pkl
    clf_pred_bml.npy         Test-set predictions and ground truth.
    clf_true_bml.npy
    Info_log.txt             Mirror of everything printed to console.
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

from dataset_clf_bml_v2 import (N_CLASSES, N_INPUT, RUL_EDGES, V_BINS,
                                build_clf_dataloaders)
from train_utils import Tee, class_names, print_model_summary, set_seed


def save_model_config(path: str, args, summary_feats: int) -> None:
    """Record the architecture next to the weights.

    The checkpoint itself stays a bare state_dict so load_state_dict keeps
    working; the shapes it must be rebuilt with live here instead of being
    retyped by hand at inference time. max_seq_len is included because it
    now sizes the positional-encoding table.

    Args:
        path: Destination .json file.
        args: Parsed command-line arguments.
        summary_feats: Width of the summary vector the scaler produced.
    """
    config = {
        "cnn_dim": args.cnn_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "summary_feats": summary_feats,
        "n_classes": N_CLASSES,
        "n_input": N_INPUT,
        "v_bins": V_BINS,
        "loss": args.loss,
        "seed": args.seed,
        "content_dir": args.content_dir,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---- model ---------------------------------------------------------------


class OrdinalLoss(nn.Module):
    """Cumulative-threshold loss over raw logits, same shape as CrossEntropy.

    Inlined rather than imported from model_clf: that module carries the
    CNN+GRU classifier, which this trainer does not use, and importing it
    for 20 lines of loss would drag the whole model in behind it.

        P(y > k) = sum_{j=k+1}^{K-1} softmax(logits)_j
    """

    def __init__(self, n_classes: int = 5, reduction: str = "mean"):
        super().__init__()
        self.K = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        cum = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]
        thresholds = torch.arange(self.K - 1, device=targets.device)
        labels = (targets.unsqueeze(1) > thresholds).float()
        loss = nn.functional.binary_cross_entropy(
            cum.clamp(1e-7, 1 - 1e-7), labels, reduction="none")
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == "mean" else loss.sum()


class BatteryRULTransformer(nn.Module):
    """CNN feature extractor per cycle, then a Transformer over the window.

    Named BatteryRULTransformer, not BatteryRULClassifier: the repo already
    had two different architectures sharing that one name across two files,
    which is how a trainer ended up importing from both and computing a
    pred_fn nothing accepted.

    Input:
        dq:      (B, N_INPUT, 1, V_BINS)  delta-Qdlin curves
        summary: (B, N_INPUT, summary_feats)

    Shape flow:
        dq      -> time-distributed Conv1d -> (B, T, cnn_dim)
        summary -> Linear                  -> (B, T, cnn_dim)
        concat, project                    -> (B, T, d_model)
        + positional encoding
        TransformerEncoder                 -> (B, T, d_model)
        mean pool, MLP head                -> (B, n_classes)
    """

    def __init__(self, cnn_dim: int = 32, d_model: int = 32, n_heads: int = 4,
                 n_enc_layers: int = 2, d_ff: int = 32, summary_feats: int = 12,
                 n_classes: int = 5, dropout: float = 0.1,
                 max_seq_len: int = 64):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(64, cnn_dim, kernel_size=5, padding=2), nn.BatchNorm1d(cnn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.summary_proj = nn.Sequential(
            nn.Linear(summary_feats, cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_proj = nn.Linear(cnn_dim + cnn_dim, d_model)

        # persistent=False keeps the table out of state_dict: it is derived
        # from (max_seq_len, d_model), not learned, so saving it would bloat
        # every checkpoint with values that can be recomputed exactly.
        self.register_buffer("pos_enc",
                             self._build_pos_enc(max_seq_len, d_model),
                             persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,          # Pre-LN, more stable than Post-LN
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal position table, shape (1, max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """Classify one batch of cycle windows.

        Args:
            dq: (B, T, 1, V_BINS) delta-Qdlin curves.
            summary: (B, T, summary_feats) per-cycle scalars.

        Returns:
            (B, n_classes) logits.

        Raises:
            ValueError: If the window is longer than the position table.
        """
        B, T, C, F = dq.shape

        dq_feat = self.cnn(dq.reshape(B * T, C, F)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused = self.dropout(torch.cat([dq_feat, summary_feat], dim=-1))
        x = self.input_proj(fused)

        # Without this, self-attention sees the window as an unordered set:
        # measured on the previous version, f(x) and f(reversed x) were equal
        # to 0.000000, so cycle order carried no information at all.
        if T > self.pos_enc.size(1):
            raise ValueError(
                f"window length {T} exceeds max_seq_len {self.pos_enc.size(1)}")
        x = x + self.pos_enc[:, :T]

        x = self.encoder(x)
        return self.head(x.mean(dim=1))


# ---- train / eval loops ---------------------------------------------------


def train_epoch(model, loader, criterion, optimizer, device) -> tuple:
    """Run one training epoch.
    
    Args:
        model: Model to train.
        loader: Training batches.
        criterion: Loss function.
        optimizer: Optimiser to step.
        device: Device to run on.

    Returns:
        ``(mean_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss = correct = total = 0
    for batch in loader:
        dq, summary = batch["dq"].to(device), batch["summary"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(dq, summary)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()


def evaluate(model, loader, criterion, device) -> tuple:
    """Score the model over a whole loader.

    Args:
        model: Model to evaluate.
        loader: Batches to score.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        ``(mean_loss, accuracy, predictions, ground_truth)``.
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for batch in loader:
        dq, summary = batch["dq"].to(device), batch["summary"].to(device)
        labels = batch["label"].to(device)

        logits = model(dq, summary)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    all_pred, all_true = np.array(all_pred), np.array(all_true)
    return total_loss / len(all_true), (all_pred == all_true).mean(), all_pred, all_true


# ---- training -------------------------------------------------------------


def train(args) -> None:
    """Run one training job end to end and write every artefact."""
    set_seed(args.seed)
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

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler,
                os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULTransformer(
        cnn_dim=args.cnn_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers, d_ff=args.d_ff,
        summary_feats=summary_feats, n_classes=N_CLASSES,
        dropout=args.dropout, max_seq_len=args.max_seq_len,
    ).to(device)
    print_model_summary(model, N_INPUT, V_BINS, summary_feats, device)
    save_model_config(os.path.join(args.output_dir, "model_config.json"),
                      args, summary_feats)

    criterion = (OrdinalLoss(n_classes=N_CLASSES) if args.loss == "ordinal"
                 else nn.CrossEntropyLoss())
    print(f"Loss: {type(criterion).__name__}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_path = os.path.join(args.output_dir, "best_clf_bml.pt")
    best_val_acc = -1.0

    header = (f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | "
              f"{'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")

        # Strictly greater: the old `>=` re-saved on every tie, so a run that
        # had stopped improving kept overwriting its own best checkpoint.
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    print("\n" + "=" * 60)
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    # labels= is required, not optional: without it sklearn infers the class
    # count from the data and raises when the test split happens to contain
    # fewer than N_CLASSES classes, which BML cells routinely cause.
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
        description="Train the CNN+Transformer RUL classifier on BML features.")
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--output_dir", default="./checkpoints_clf_bml_tf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cnn_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_enc_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="Length of the positional-encoding table; must "
                             "be at least N_INPUT.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", default="ordinal",
                        choices=["cross_entropy", "ordinal"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
