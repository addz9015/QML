"""Shared building blocks for the Phase 3 (noise) and Phase 4 (mitigation) studies.

Protocol used everywhere, so that every number in the report is comparable:

* Data     : ResNet features of data/train (fit) and data/test (held-out, never used
             for fitting, model selection or calibration).
* Seeds    : experiment cell = one seed s in range(CFG.n_seeds). Seed s fixes the
             80/20 train/val split (inside data/train), the scaler and the model init.
* Ideal    : the model trained on the noise-free circuit ("ideal-trained"), cached to
             models/cache so every study evaluates the very same weights.
* Noise    : one channel after every executed layer on every qubit, exact density-matrix
             simulation (no shot noise), so the only randomness is training.
* Reporting: mean and sample std (ddof=1) over seeds, plus paired differences.
"""

import os
import sys
import csv

for _stream in (sys.stdout, sys.stderr):                 # the Windows console defaults to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from model import HybridModel, fit, to_tensor            # imports pennylane before scipy/sklearn
from quantum_model import make_qnode, run_qnode, NOISE_TYPES

import numpy as np
import torch
from scipy import stats as sps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from train import get_train_features, get_test_features

RAW_FIELDS = ["stage", "method", "depth", "entanglement", "noise_type", "p", "seed",
              "accuracy", "roc_auc", "mean_abs_z", "frac_pred_class1", "rmse", "epochs"]
_NUMERIC = {"depth": int, "p": float, "seed": int, "accuracy": float, "roc_auc": float,
            "mean_abs_z": float, "frac_pred_class1": float, "rmse": float, "epochs": int}
KEY_FIELDS = ("stage", "method", "depth", "entanglement", "noise_type", "p", "seed")


# ---------------------------------------------------------------------------- noise
def noise_spec(kind, p):
    """{"type", "p"} for p > 0, else None (ideal circuit)."""
    return {"type": kind, "p": float(p)} if p > 0 else None


# ---------------------------------------------------------------------------- data
class Data:
    """Raw (unscaled) ResNet features for the train and held-out test images."""

    def __init__(self):
        self.feats, self.labels, self.classes = get_train_features()
        self.t_feats, self.t_labels, _ = get_test_features()


def seed_split(data, seed):
    """Seed-specific 80/20 split of data/train and a scaler fit on its train part."""
    tr, va = train_test_split(np.arange(len(data.labels)), test_size=0.2,
                              stratify=data.labels, random_state=CFG.seed + seed)
    sc = StandardScaler().fit(data.feats[tr])
    return {
        "sc": sc,
        "Xtr": sc.transform(data.feats[tr]), "ytr": data.labels[tr],
        "Xva": sc.transform(data.feats[va]), "yva": data.labels[va],
        "Xte": sc.transform(data.t_feats), "yte": data.t_labels,
    }


# --------------------------------------------------------------------------- models
def build_model(in_dim, depth=None, entanglement=None, noise=None, use_circuit=True):
    return HybridModel(in_dim=in_dim, n_qubits=CFG.n_qubits,
                       n_layers=CFG.n_layers if depth is None else depth,
                       use_circuit=use_circuit, noise=noise,
                       entanglement=CFG.entanglement if entanglement is None else entanglement)


def train_model(split, seed, depth=None, entanglement=None, noise=None, epochs=None, lr=None,
                weight_decay=None):
    """Train from scratch on split["Xtr"]; the best-validation epoch is kept."""
    torch.manual_seed(CFG.seed + seed)
    model = build_model(split["Xtr"].shape[1], depth, entanglement, noise)
    fit(model, to_tensor(split["Xtr"]), to_tensor(split["ytr"], torch.long),
        to_tensor(split["Xva"]), to_tensor(split["yva"], torch.long),
        epochs=epochs or CFG.epochs, lr=lr or CFG.lr,
        weight_decay=CFG.weight_decay if weight_decay is None else weight_decay,
        seed=CFG.seed + seed, verbose=False)
    model.eval()
    return model


def _ideal_path(depth, entanglement, seed):
    # the key includes the training recipe, so changing it never reuses stale weights
    return os.path.join(CFG.cache_dir, f"ideal_d{depth}_{entanglement}_s{seed}"
                                       f"_e{CFG.epochs}_lr{CFG.lr:g}_wd{CFG.weight_decay:g}.pt")


def get_ideal(data, seed, depth=None, entanglement=None):
    """Ideal-circuit model for (design, seed): loaded from cache or trained once.

    Returns (model, split). Every study calls this, so they all share the same weights.
    """
    depth = CFG.n_layers if depth is None else depth
    entanglement = CFG.entanglement if entanglement is None else entanglement
    split = seed_split(data, seed)
    path = _ideal_path(depth, entanglement, seed)
    if os.path.exists(path):
        model = build_model(split["Xtr"].shape[1], depth, entanglement)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
    else:
        model = train_model(split, seed, depth, entanglement)
        os.makedirs(CFG.cache_dir, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model, split


# ------------------------------------------------------------------ circuit outputs
@torch.no_grad()
def qfeatures(model, X, noise=None, fold=0):
    """Quantum features <Z_i> of `model`'s circuit on scaled inputs X, under `noise`.

    `fold` amplifies the noise by gate folding (see quantum_model). The projection and
    BatchNorm run in eval mode, so only the circuit differs from the ideal model.
    """
    model.eval()
    z = model.encode(to_tensor(X))
    qnode = make_qnode(model.n_qubits, model.n_layers, noise, model.entanglement, fold)
    return run_qnode(qnode, z, model.qlayer.weights.detach())


def head_proba(model, q):
    """Class probabilities from quantum features with the model's own linear readout."""
    W = model.head.weight.detach().numpy().astype(float)
    b = model.head.bias.detach().numpy().astype(float)
    logits = q @ W.T + b
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def score(proba, y, q=None):
    pred = proba.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, proba[:, 1])),
        "frac_pred_class1": float((pred == 1).mean()),
        "mean_abs_z": float(np.abs(q).mean()) if q is not None else float("nan"),
    }


def row(stage, method, seed, noise_type="", p=0.0, depth=None, entanglement=None,
        epochs=0, rmse=float("nan"), **metrics):
    return {"stage": stage, "method": method, "depth": CFG.n_layers if depth is None else depth,
            "entanglement": CFG.entanglement if entanglement is None else entanglement,
            "noise_type": noise_type, "p": round(float(p), 6), "seed": seed,
            "epochs": epochs, "rmse": rmse, **metrics}


# -------------------------------------------------------------------------- storage
def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, cast in _NUMERIC.items():
            r[k] = cast(r[k]) if r.get(k, "") not in ("", None) else (float("nan") if cast is float else 0)
    return rows


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_FIELDS})


def upsert_rows(path, new_rows):
    """Replace rows with the same key, keep every other row."""
    def key(r):
        return tuple(r[k] if k != "p" else round(float(r[k]), 6) for k in KEY_FIELDS)
    merged = {key(r): r for r in load_rows(path)}
    merged.update({key(r): r for r in new_rows})
    save_rows(path, list(merged.values()))


# ------------------------------------------------------------------------ statistics
def select(rows, **conds):
    return [r for r in rows if all(r[k] == v for k, v in conds.items())]


def by_seed(rows, field):
    """{seed: value}"""
    return {r["seed"]: r[field] for r in rows}


def mean_sd(values):
    v = np.asarray(list(values), dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired(a_by_seed, b_by_seed):
    """Paired comparison b - a over the seeds both have: (mean diff, 95% CI half-width, wins, n)."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([b_by_seed[s] - a_by_seed[s] for s in seeds], dtype=float)
    n = len(d)
    if n < 2:
        return float(d.mean()) if n else float("nan"), float("nan"), int((d > 0).sum()), n
    half = float(sps.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return float(d.mean()), half, int((d > 0).sum()), n
