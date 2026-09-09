"""train_utils.py - Setup shared by the trainers in this repo.

Four things a training run may need before it can start, none of which
depend on which architecture is being trained: seeding, console/file
logging, class naming, and a layer-by-layer model summary.

``set_seed`` is the exception to "shared": only the CMA-ES trainer calls
it. The gradient trainers deliberately leave weight init and cuDNN
process-random so that repeating a config produces a spread to average
over -- see their ``train`` docstrings.

They lived in three copies before, byte-identical in code but each with
its own docstring. That is exactly how the cuDNN determinism fix nearly
went wrong: measuring that ``torch.manual_seed`` alone was not enough on
GPU produced a two-line change that then had to be remembered in three
separate files. One copy, one place to fix.

Deliberately NOT collected here:
    save_model_config   91% identical between trainers, but the differing
                        part is the architecture field list itself
                        (gru_dim/gru_layers vs d_model/n_heads/...), a real
                        difference that sharing would only paper over.
    accuracy            98% identical, but 9 lines, and the CMA-ES trainer
                        needs a zero-guard the others do not.
    train / main        Genuinely different programs.
"""

import random
import sys

import numpy as np
import torch
import torch.nn as nn


class Tee:
    """Mirror writes to multiple streams (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed: int) -> None:
    """Seed every generator a training run draws from, including cuDNN.

    The two cudnn lines are not optional on GPU: measured on this project,
    ``torch.manual_seed`` alone still let validation loss drift between
    otherwise identical runs (0.8858 / 0.8850 / 0.8846). Pinning cuDNN to
    deterministic algorithms is what makes two runs match digit for digit.

    Args:
        seed: Base seed, normally shared with the dataloaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_names(rul_edges) -> list:
    """Readable name per RUL class, derived from the band edges.

    Deriving rather than hard-coding keeps the names honest: the original
    hand-written list ended in "RUL<100" while the labelling function uses
    ``<= 100``, so a cell with exactly 100 cycles left was named wrong in
    every report it appeared in.

    Args:
        rul_edges: Inclusive upper edge of every band but the last,
            healthiest first -- ``dataset_clf_bml_v2.RUL_EDGES``.

    Returns:
        One label per class, ordered by class index.
    """
    names = [f"RUL>{rul_edges[0]}"]
    for i in range(len(rul_edges) - 1):
        names.append(f"{rul_edges[i + 1]}<RUL<={rul_edges[i]}")
    names.append(f"RUL<={rul_edges[-1]}")
    return names


def print_model_summary(model: nn.Module, n_input: int, v_bins: int,
                        summary_feats: int, device: torch.device) -> None:
    """Print one row per leaf module by running a dummy batch through it.

    Architecture-agnostic: it only needs a model that accepts the project's
    ``(dq, summary)`` pair, so the same function describes the CNN+GRU and
    the CNN+Transformer alike.

    Hooks are registered on leaf modules only, rather than on every module
    and filtered inside the hook -- the difference is 4 wasted registrations
    on the CNN+GRU model (28 modules, 24 leaf), growing with nesting depth.

    Args:
        model: The model to describe.
        n_input: Cycles per sample (window length).
        v_bins: Length of the dq curve.
        summary_feats: Width of the summary vector.
        device: Device the dummy batch is built on.
    """
    rows, handles = [], []

    def hook(module, _inp, out):
        rows.append((module.__class__.__name__,
                     tuple(out.shape) if isinstance(out, torch.Tensor) else "?",
                     sum(p.numel() for p in module.parameters())))

    for module in model.modules():
        if not list(module.children()):          # leaf modules only
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(torch.zeros(1, n_input, 1, v_bins, device=device),
              torch.zeros(1, n_input, summary_feats, device=device))
    for handle in handles:
        handle.remove()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Summary:")
    print("=" * 80)
    print(f"  {'Layer':<38} {'Output Shape':<25} {'Params':>10}")
    print("-" * 80)
    for name, shape, n in rows:
        print(f"  {name:<38} {str(shape):<25} {n:>10,}")
    print("=" * 80)
    print(f"  {'Total trainable parameters':<38} {'':25} {total:>10,}")
    print("=" * 80 + "\n")
