from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):

    def __init__(self, in_dim, width, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_mlp(Xtr, ytr, Xva, yva, Xte, yte, *, width, epochs, seed,
              lr=1e-3, weight_decay=1e-4, batch_size=128, threshold=0.9,
              select_best_val=True):
    scaler = StandardScaler().fit(Xtr)
    to_t = lambda A: torch.from_numpy(scaler.transform(A).astype(np.float32))
    Xtr_t, Xva_t, Xte_t = to_t(Xtr), to_t(Xva), to_t(Xte)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    yva_t = torch.from_numpy(yva.astype(np.int64))
    yte_t = torch.from_numpy(yte.astype(np.int64))

    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], width, int(ytr.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n = Xtr_t.shape[0]
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, epochs, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist["train_acc"].append(_accuracy(model(Xtr_t), ytr_t))
            va = _accuracy(model(Xva_t), yva_t)
        hist["val_acc"].append(va)
        if select_best_val and va > best_val_acc:
            best_val_acc, best_epoch = va, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if select_best_val and best_state is not None:
        model.load_state_dict(best_state)  # restore the best-validation checkpoint
    else:
        best_val_acc, best_epoch = hist["val_acc"][-1], epochs

    model.eval()
    with torch.no_grad():
        te_logits = model(Xte_t)
    val = np.asarray(hist["val_acc"])
    reached = np.where(val >= threshold)[0]
    return {
        "hist": hist,
        "test_acc": _accuracy(te_logits, yte_t),
        "test_pred": te_logits.argmax(1).cpu().numpy(),
        "ep_to_thr": int(reached[0]) + 1 if reached.size else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "params": count_params(model),
        "model": model,      # best-val checkpoint; reusable on new inputs (rotations)
        "scaler": scaler,
    }


def default_grid():
    return {"lr": [3e-4, 1e-3, 3e-3], "weight_decay": [0.0, 1e-4, 1e-3]}


def _grid_configs(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune_head(Xtr, ytr, Xva, yva, Xte, yte, *, width, epochs, seed, grid=None):
    grid = default_grid() if grid is None else grid
    best = None
    for combo in _grid_configs(grid):
        r = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, width=width, epochs=epochs,
                      seed=seed, select_best_val=True, **combo)
        r["hp"] = {"lr": combo.get("lr", 1e-3),
                   "weight_decay": combo.get("weight_decay", 1e-4), "width": width}
        if best is None or r["best_val_acc"] > best["best_val_acc"]:
            best = r
    return best


def predict(model, scaler, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(scaler.transform(X).astype(np.float32))
        return model(Xt).argmax(1).cpu().numpy()


def _interp_threshold(xs, accs, threshold):
    for i, a in enumerate(accs):
        if a >= threshold:
            if i == 0 or accs[i - 1] >= a:
                return float(xs[i])
            frac = min(max((threshold - accs[i - 1]) / (a - accs[i - 1]), 0.0), 1.0)
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return None


def params_to_threshold(cap_rep, threshold):
    p = _interp_threshold(cap_rep["params"], cap_rep["mean"], threshold)
    return None if p is None else int(round(p))


def width_to_threshold(cap_rep, threshold):
    return _interp_threshold(cap_rep["widths"], cap_rep["mean"], threshold)
