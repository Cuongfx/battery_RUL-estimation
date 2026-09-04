"""model_clf.py - CNN+GRU classifier, ordinal loss, and the train/eval loops.

Model definitions only. This module deliberately imports no dataset or
feature-export module, so pulling the model into a trainer never drags a
different pipeline's data layer along with it. Every trainer builds its own
dataloaders and passes the model what it needs.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """
    Takes raw logits (B, 5) — same as CrossEntropyLoss.
    Converts to cumulative probabilities internally using softmax.

    P(y > k) = sum_{j=k+1}^{K-1} softmax(logits)_j
    """
    def __init__(self, n_classes: int = 5, reduction: str = 'mean'):
        super().__init__()
        self.K         = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)                          # (B, 5)
        cum   = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]             # (B, K-1)
        thresholds = torch.arange(self.K - 1, device=targets.device)
        labels     = (targets.unsqueeze(1) > thresholds).float()     # (B, K-1)
        loss = F.binary_cross_entropy(cum.clamp(1e-7, 1 - 1e-7), labels, reduction='none')
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def predict_cls(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1)


def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    cum   = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]   # (B, 4)
    return (cum > 0.5).sum(dim=1).long()                # (B,)


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class BatteryRULClassifier(nn.Module):
    def __init__(
        self,
        cnn_dim:       int   = 32,
        gru_dim:       int   = 32,
        gru_layers:    int   = 2,
        summary_feats: int   = 16,
        # A literal, not dataset_clf.N_CLASSES: this module must not
        # import a dataset module. Every caller passes n_classes
        # explicitly, so this default is never exercised.
        n_classes:     int   = 5,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16),  nn.ELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_dim, kernel_size=5, padding=2), nn.BatchNorm1d(cnn_dim), nn.ELU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )


        self.summary_proj = nn.Sequential(
            nn.Linear(summary_feats, cnn_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size    = cnn_dim + cnn_dim,      # 2 channels × cnn_dim each + summary proj cnn_dim
            hidden_size   = gru_dim,
            num_layers    = gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if gru_layers > 1 else 0.0,
        )

        self.post_gru_drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(gru_dim * 2, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        B, T, C, F = dq.shape                                       

        dq_feat      = self.cnn(dq.reshape(B * T, C, F)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused    = self.post_gru_drop(torch.cat([dq_feat, summary_feat], dim=-1))
        out, h_n = self.gru(fused)
        h_last   = self.post_gru_drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))

        return self.head(h_last)


# -------------------------------------------------------------------------
# Train one epoch
# -------------------------------------------------------------------------
def train_epoch(model, loader, criterion, pred_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(dq, summary)
        loss = criterion(out, labels)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (pred_fn(out) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, pred_fn, device):
    model.eval()
    total_loss = 0.0
    all_pred   = []
    all_true   = []
    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)
        out     = model(dq, summary)
        loss    = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        all_pred.extend(pred_fn(out).cpu().numpy())
        all_true.extend(labels.cpu().numpy())
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    return total_loss / len(all_true), (all_pred == all_true).mean(), all_pred, all_true
