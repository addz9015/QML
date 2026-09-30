"""Honest evaluation: stratified k-fold CV + held-out test, with plots.

    python -m src.evaluate                 # circuit = trained
    python -m src.evaluate --circuit none  # classical baseline
    python -m src.evaluate --circuit fixed # frozen random circuit

Writes results/metrics.csv rows and results/{confusion,roc}_<circuit>.png.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from model import HybridModel, fit, to_tensor

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve,
)

from train import get_train_features, get_test_features
from tracking import log_run


def build(in_dim, circuit):
    return HybridModel(in_dim=in_dim, n_qubits=CFG.n_qubits, n_layers=CFG.n_layers,
                       freeze_circuit=(circuit == "fixed"),
                       use_circuit=(circuit != "none"),
                       entanglement=CFG.entanglement)


def train_eval(circuit, Xtr, ytr, Xte, yte, epochs, seed=CFG.seed):
    """Fit on (Xtr, ytr) for a fixed number of epochs, return P(class 1) for Xte."""
    torch.manual_seed(seed)
    model = build(Xtr.shape[1], circuit)
    fit(model, to_tensor(Xtr), to_tensor(ytr, torch.long),
        epochs=epochs, lr=CFG.lr, weight_decay=CFG.weight_decay,
        seed=seed, verbose=False)
    return model.predict_proba(to_tensor(Xte))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=["trained", "fixed", "none"], default="trained")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    args = ap.parse_args()
    os.makedirs(CFG.results_dir, exist_ok=True)

    feats, labels, classes = get_train_features()
    print(f"{len(labels)} train images | circuit={args.circuit}")

    # ---------- stratified k-fold CV (scaler fit per fold) ----------
    skf = StratifiedKFold(n_splits=CFG.cv_folds, shuffle=True, random_state=CFG.seed)
    oof_prob = np.zeros(len(labels))
    fold_acc = []
    for k, (tr, va) in enumerate(skf.split(feats, labels), 1):
        sc = StandardScaler().fit(feats[tr])
        p = train_eval(args.circuit, sc.transform(feats[tr]), labels[tr],
                       sc.transform(feats[va]), labels[va], args.epochs)
        oof_prob[va] = p
        fold_acc.append(accuracy_score(labels[va], (p > 0.5).astype(int)))
        print(f"  fold {k}: acc={fold_acc[-1]:.3f}")

    cv_acc, cv_std = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=1))
    cv_auc = roc_auc_score(labels, oof_prob)
    print(f"\nCV accuracy : {cv_acc:.3f} +/- {cv_std:.3f}")
    print(f"CV ROC-AUC  : {cv_auc:.3f}")
    log_run(phase="1-2", circuit=args.circuit, eval="cv",
            n_qubits=CFG.n_qubits, n_layers=CFG.n_layers, n_train=len(labels),
            accuracy=round(cv_acc, 4), accuracy_std=round(cv_std, 4),
            roc_auc=round(cv_auc, 4))

    # ---------- held-out test: CFG.n_seeds independent trainings on all training data ----------
    sc = StandardScaler().fit(feats)
    t_raw, t_labels, _ = get_test_features()
    t_feat = sc.transform(t_raw)
    accs, aucs, probs = [], [], []
    for s in range(CFG.n_seeds):
        pr = train_eval(args.circuit, sc.transform(feats), labels, t_feat, t_labels,
                        args.epochs, seed=CFG.seed + s)
        probs.append(pr)
        accs.append(accuracy_score(t_labels, (pr > 0.5).astype(int)))
        aucs.append(roc_auc_score(t_labels, pr))
    test_acc, test_sd = float(np.mean(accs)), float(np.std(accs, ddof=1))
    test_auc = float(np.mean(aucs))
    test_prob = probs[0]                                  # the plots below show the first run
    test_pred = (test_prob > 0.5).astype(int)
    print(f"\nheld-out test accuracy : {test_acc:.3f} +/- {test_sd:.3f}  ({CFG.n_seeds} trainings)")
    print(f"held-out test ROC-AUC  : {test_auc:.3f}")
    log_run(phase="1-2", circuit=args.circuit, eval="test",
            n_qubits=CFG.n_qubits, n_layers=CFG.n_layers, n_train=len(labels),
            accuracy=round(test_acc, 4), accuracy_std=round(test_sd, 4),
            roc_auc=round(test_auc, 4))

    # ---------- plots ----------
    cm = confusion_matrix(t_labels, test_pred)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], classes); ax.set_yticks([0, 1], classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion — circuit={args.circuit}")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, f"confusion_{args.circuit}.png"), dpi=130)

    fpr, tpr, _ = roc_curve(t_labels, test_prob)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC — circuit={args.circuit}"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, f"roc_{args.circuit}.png"), dpi=130)
    print("plots ->", CFG.results_dir)


if __name__ == "__main__":
    main()
