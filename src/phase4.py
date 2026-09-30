"""Phase 4 - mitigating the effect of noise.

Phase 3 showed that noise mainly shrinks the quantum outputs <Z> towards 0 and
shifts the decision threshold, while ROC-AUC stays high. Three ways to respond:

  recal  : keep the (ideal-trained) circuit, re-fit only the classical readout on
           noisy features:  "threshold" (1 parameter) or "head-refit" (full linear
           readout with feature re-scaling). Needs a few noisy calibration runs.
  zne    : zero-noise extrapolation of the circuit OUTPUTS <Z_i> (a smooth
           observable - not of accuracy), then the ideal readout.
             scale : noise strength p is amplified to 2p, 3p (simulator stand-in)
             fold  : gate folding L -> L (L^+ L)^k, noise amplified 1, 3, 5x
                     (what is done on real hardware; needed for Phase 5)
           each with a linear (2-point) or Richardson (3-point) extrapolation.
  design : circuit depth (1-3) x entanglement (ring / linear / none), every design
           trained from scratch with the same budget and the same seeds; evaluated
           unmitigated and with a re-fitted readout.

The full noise-aware retraining (Phase 3, --stage aware) is the expensive reference.
All methods use the same models, seeds, test set and metrics (see experiments.py).

    python -m src.phase4 --stage recal
    python -m src.phase4 --stage zne
    python -m src.phase4 --stage design
    python -m src.phase4 --stage report
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (CFG, NOISE_TYPES, Data, get_ideal, qfeatures, head_proba, score, row,
                         noise_spec, load_rows, upsert_rows, select, by_seed, mean_sd, paired)
from quantum_model import ENTANGLEMENTS
from tracking import log_run
import noise_study

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = os.path.join(CFG.results_dir, "phase4_raw.csv")
P_GRID = [0.05, 0.1, 0.15, 0.2, 0.3]
DESIGN_P = [0.0, 0.05, 0.1, 0.15, 0.2]
DEPTHS = (1, 2, 3)
SCALE_LAMS = (1.0, 2.0, 3.0)      # noise-strength amplification factors
FOLD_LAMS = (1.0, 3.0, 5.0)       # 1 + 2k  <-  k = 0, 1, 2 folds per layer
COLORS = noise_study.COLORS
LABEL = noise_study.LABEL


# ------------------------------------------------------------------------ mitigation
def fit_threshold(scores, y):
    """Decision threshold on a 1-D score that maximises calibration accuracy.

    Ties are broken towards 0 (the original decision boundary), so an already
    well-calibrated readout is left unchanged.
    """
    s = np.sort(scores)
    cands = np.concatenate([[s[0] - 1.0], (s[:-1] + s[1:]) / 2.0, [s[-1] + 1.0]])
    accs = np.array([((scores > t).astype(int) == y).mean() for t in cands])
    best = cands[accs == accs.max()]
    return float(best[np.argmin(np.abs(best))])


def threshold_proba(model, q_cal, y_cal, q_test):
    """Ideal readout with its decision threshold re-fitted on noisy calibration data."""
    def margin(q):
        W = model.head.weight.detach().numpy().astype(float)
        b = model.head.bias.detach().numpy().astype(float)
        z = q @ W.T + b
        return z[:, 1] - z[:, 0]
    t = fit_threshold(margin(q_cal), y_cal)
    p1 = 1.0 / (1.0 + np.exp(-(margin(q_test) - t)))
    return np.stack([1.0 - p1, p1], axis=1)


def head_refit_proba(q_cal, y_cal, q_test):
    """Re-fit the whole linear readout (with feature re-scaling) on noisy calibration data."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(q_cal, y_cal)
    return clf.predict_proba(q_test)


def extrapolate_to_zero(lams, values, order):
    """Lagrange extrapolation of noisy expectation values to noise factor 0.

    order 1: line through the first two points; order 2: Richardson through all three.
    Result is clipped to the physical range [-1, 1] of <Z>.
    """
    n = order + 1
    lams = np.asarray(lams[:n], float)
    out = 0.0
    for i in range(n):
        w = np.prod([(0.0 - lams[j]) / (lams[i] - lams[j]) for j in range(n) if j != i])
        out = out + w * values[i]
    return np.clip(out, -1.0, 1.0)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------- experiments
def run_recal_zne(n_seeds, p_grid, do_recal=True, do_zne=True):
    data = Data()
    for seed in range(n_seeds):
        t0 = time.time()
        model, split = get_ideal(data, seed)
        Xcal = np.concatenate([split["Xtr"], split["Xva"]])      # training-side data only
        ycal = np.concatenate([split["ytr"], split["yva"]])
        yte = split["yte"]
        q_ideal = qfeatures(model, split["Xte"])
        rows = []
        for kind in NOISE_TYPES:
            for p in p_grid:
                noise = noise_spec(kind, p)
                q_te = qfeatures(model, split["Xte"], noise)
                base = score(head_proba(model, q_te), yte, q_te)
                rows.append(row("phase4", "none", seed, kind, p, rmse=rmse(q_te, q_ideal), **base))

                if do_recal:
                    q_cal = qfeatures(model, Xcal, noise)
                    for name, proba in (("threshold", threshold_proba(model, q_cal, ycal, q_te)),
                                        ("head-refit", head_refit_proba(q_cal, ycal, q_te))):
                        rows.append(row("phase4", name, seed, kind, p, **score(proba, yte, q_te)))

                if do_zne:
                    sets = {
                        "scale": [q_te] + [qfeatures(model, split["Xte"], noise_spec(kind, p * lam))
                                           for lam in SCALE_LAMS[1:]],
                        "fold": [q_te] + [qfeatures(model, split["Xte"], noise, fold=int((lam - 1) // 2))
                                          for lam in FOLD_LAMS[1:]],
                    }
                    for how, lams in (("scale", SCALE_LAMS), ("fold", FOLD_LAMS)):
                        for order, oname in ((1, "lin"), (2, "rich")):
                            q_m = extrapolate_to_zero(lams, sets[how], order)
                            rows.append(row("phase4", f"zne-{how}-{oname}", seed, kind, p,
                                            rmse=rmse(q_m, q_ideal),
                                            **score(head_proba(model, q_m), yte, q_m)))
        upsert_rows(RAW, rows)
        print(f"  seed {seed}: done ({time.time() - t0:.0f}s)", flush=True)


def run_design(n_seeds, p_grid):
    data = Data()
    for depth in DEPTHS:
        for ent in ENTANGLEMENTS:
            for seed in range(n_seeds):
                t0 = time.time()
                model, split = get_ideal(data, seed, depth, ent)      # trains once, then cached
                Xcal = np.concatenate([split["Xtr"], split["Xva"]])
                ycal = np.concatenate([split["ytr"], split["yva"]])
                rows = []
                for kind in NOISE_TYPES:
                    for p in p_grid:
                        noise = noise_spec(kind, p)
                        q_te = qfeatures(model, split["Xte"], noise)
                        common = dict(depth=depth, entanglement=ent)
                        rows.append(row("design", "none", seed, kind, p, **common,
                                        **score(head_proba(model, q_te), split["yte"], q_te)))
                        if p > 0:
                            q_cal = qfeatures(model, Xcal, noise)
                            rows.append(row("design", "head-refit", seed, kind, p, **common,
                                            **score(head_refit_proba(q_cal, ycal, q_te),
                                                    split["yte"], q_te)))
                upsert_rows(RAW, rows)
                print(f"  depth={depth} {ent:6s} seed={seed}: done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------- report
MITIGATIONS = [("none", "no mitigation", "o-", "#444444"),
               ("threshold", "re-fit threshold", "s-", "#1f77b4"),
               ("head-refit", "re-fit readout", "^-", "#2ca02c"),
               ("zne-fold-rich", "ZNE (gate folding)", "v-", "#d62728"),
               ("zne-scale-rich", "ZNE (noise scaling)", "D--", "#ff7f0e")]


def _series(rows3, rows4, method, kind, field, ps):
    """mean, sd over seeds at each p for one method (p = 0 comes from the noise-free model)."""
    means, sds = [], []
    for p in ps:
        if p == 0.0:
            sel = select(rows3, method="ideal-trained", noise_type=kind, p=0.0)
        elif method == "aware":
            sel = select(rows3, method="aware", noise_type=kind, p=p)
        else:
            sel = select(rows4, stage="phase4", method=method, noise_type=kind, p=p)
        if sel:
            m, s = mean_sd(r[field] for r in sel)
        else:
            m = s = float("nan")
        means.append(m)
        sds.append(s)
    return np.array(means), np.array(sds)


def make_report():
    rows4 = load_rows(RAW)
    rows3 = load_rows(noise_study.RAW)
    if not rows4 or not rows3:
        raise SystemExit("Run the Phase 3 inference stage and the Phase 4 stages first.")
    os.makedirs(CFG.results_dir, exist_ok=True)
    main = [r for r in rows4 if r["stage"] == "phase4"]
    n_seeds = len({r["seed"] for r in main})
    ps_all = [0.0] + P_GRID

    # --- consistency check: the "none" baseline here must equal Phase 3's ideal-trained numbers
    worst = 0.0
    for r in select(main, method="none"):
        m = select(rows3, method="ideal-trained", noise_type=r["noise_type"], p=r["p"], seed=r["seed"])
        if m:
            worst = max(worst, abs(m[0]["accuracy"] - r["accuracy"]))
    print(f"consistency check (Phase 4 'none' vs Phase 3 'ideal-trained'): max |diff| = {worst:.4f}")

    # --- Figure: accuracy vs p for every mitigation
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, kind in zip(axes, NOISE_TYPES):
        for method, label, style, color in MITIGATIONS:
            m, s = _series(rows3, main, method, kind, "accuracy", ps_all)
            ax.plot(ps_all, m, style, color=color, label=label, ms=4, lw=1.4)
            ax.fill_between(ps_all, m - s, m + s, color=color, alpha=0.08)
        pa = [p for p in noise_study.AWARE_P]
        m, s = _series(rows3, main, "aware", kind, "accuracy", pa)
        ax.errorbar(pa, m, yerr=s, fmt="*", color="black", ms=9, capsize=3, label="noise-aware retraining")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(LABEL[kind])
        ax.set_xlabel("noise probability p")
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"Phase 4: accuracy under noise with each mitigation (mean ± sd over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, "phase4_mitigation.png"), dpi=140)
    plt.close(fig)

    # --- Figure: ZNE quality on the observable itself
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    zne_methods = [("none", "unmitigated", "o-", "#444444"),
                   ("zne-scale-lin", "scaling, linear", "D:", "#ffbb78"),
                   ("zne-scale-rich", "scaling, Richardson", "D--", "#ff7f0e"),
                   ("zne-fold-lin", "folding, linear", "v:", "#ff9896"),
                   ("zne-fold-rich", "folding, Richardson", "v-", "#d62728")]
    for ax, kind in zip(axes, NOISE_TYPES):
        for method, label, style, color in zne_methods:
            m, _ = _series(rows3, main, method, kind, "rmse", P_GRID)
            ax.plot(P_GRID, m, style, color=color, label=label, ms=4)
        ax.set_yscale("log")
        ax.set_title(LABEL[kind])
        ax.set_xlabel("noise probability p")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMSE of <Z> vs noise-free value")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Phase 4: how close is the (mitigated) quantum output to the noise-free output?")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, "phase4_zne.png"), dpi=140)
    plt.close(fig)

    lines = [f"Held-out test set: {2 * CFG.test_per_class} images; mean ± sd over {n_seeds} seeds. "
             "'vs none' = paired difference in accuracy (mean ± 95% CI, seeds where the method is better).",
             f"Consistency check vs Phase 3 baseline: max |diff| = {worst:.4f}.", ""]

    # ---- mitigation table
    header = ["noise", "p"] + [m[1] for m in MITIGATIONS] + ["noise-aware retraining"]
    lines += ["### Accuracy under noise", "", "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
    for kind in NOISE_TYPES:
        for p in P_GRID:
            base = select(main, method="none", noise_type=kind, p=p)
            cells = [LABEL[kind], f"{p:g}"]
            for method, *_ in MITIGATIONS + [("aware",)]:
                sel = (select(rows3, method="aware", noise_type=kind, p=p) if method == "aware"
                       else select(main, method=method, noise_type=kind, p=p))
                if not sel:
                    cells.append("-")
                    continue
                a, sd = mean_sd(r["accuracy"] for r in sel)
                txt = f"{a:.3f} ± {sd:.3f}"
                if method != "none":
                    d, ci, wins, n = paired(by_seed(base, "accuracy"), by_seed(sel, "accuracy"))
                    txt += f" ({d:+.3f}±{ci:.3f}; {wins}/{n})"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")

    # ---- ZNE table
    lines += ["", "### Zero-noise extrapolation: RMSE of <Z> against the noise-free value", "",
              "| noise | p | unmitigated | scaling, linear | scaling, Richardson | folding, linear | folding, Richardson |",
              "|---|---|---|---|---|---|---|"]
    for kind in NOISE_TYPES:
        for p in P_GRID:
            cells = [LABEL[kind], f"{p:g}"]
            for method in ("none", "zne-scale-lin", "zne-scale-rich", "zne-fold-lin", "zne-fold-rich"):
                sel = select(main, method=method, noise_type=kind, p=p)
                cells.append(f"{mean_sd(r['rmse'] for r in sel)[0]:.3f}" if sel else "-")
            lines.append("| " + " | ".join(cells) + " |")

    # ---- design sweep
    design = [r for r in rows4 if r["stage"] == "design"]
    if design:
        def cell(depth, ent, method, p):
            """per-seed accuracy averaged over the three noise types -> mean, sd over seeds"""
            per_seed = []
            for seed in sorted({r["seed"] for r in design}):
                sel = select(design, depth=depth, entanglement=ent, method=method, p=p, seed=seed)
                if sel:
                    per_seed.append(np.mean([r["accuracy"] for r in sel]))
            return mean_sd(per_seed) if per_seed else (float("nan"), float("nan"))

        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
        panels = [("none", 0.0, "noise-free"), ("none", 0.15, "noise p = 0.15, no mitigation"),
                  ("head-refit", 0.15, "noise p = 0.15, re-fitted readout")]
        for ax, (method, p, title) in zip(axes, panels):
            grid = np.array([[cell(d, e, method, p)[0] for e in ENTANGLEMENTS] for d in DEPTHS])
            im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(ENTANGLEMENTS)), ENTANGLEMENTS)
            ax.set_yticks(range(len(DEPTHS)),
                          [f"{d} layer{'s' if d > 1 else ''}" for d in DEPTHS] if ax is axes[0] else [])
            for i in range(len(DEPTHS)):
                for j in range(len(ENTANGLEMENTS)):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="white" if grid[i, j] < 0.8 else "black")
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axes, fraction=0.02, label="test accuracy")
        fig.suptitle(f"Phase 4: circuit design (accuracy averaged over the 3 noise types and {n_seeds} seeds)")
        fig.savefig(os.path.join(CFG.results_dir, "phase4_design.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)

        lines += ["", "### Circuit design (accuracy averaged over the 3 noise types; mean ± sd over seeds)", "",
                  "| depth | entanglement | noise-free | p=0.1 | p=0.2 | p=0.2, re-fitted readout |",
                  "|---|---|---|---|---|---|"]
        for d in DEPTHS:
            for e in ENTANGLEMENTS:
                vals = [cell(d, e, "none", 0.0), cell(d, e, "none", 0.1), cell(d, e, "none", 0.2),
                        cell(d, e, "head-refit", 0.2)]
                lines.append(f"| {d} | {e} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in vals) + " |")

    text = "\n".join(lines)
    with open(os.path.join(CFG.results_dir, "phase4_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)

    for method in sorted({r["method"] for r in main}):
        for kind in NOISE_TYPES:
            for p in P_GRID:
                sel = select(main, method=method, noise_type=kind, p=p)
                if sel:
                    a, sd = mean_sd(r["accuracy"] for r in sel)
                    auc, _ = mean_sd(r["roc_auc"] for r in sel)
                    log_run(phase="4", circuit=method, noise_type=kind, noise_p=p,
                            n_qubits=CFG.n_qubits, n_layers=CFG.n_layers, eval="test",
                            accuracy=round(a, 4), accuracy_std=round(sd, 4), roc_auc=round(auc, 4))


# ------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["recal", "zne", "design", "report", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=CFG.n_seeds)
    ap.add_argument("--p-grid", type=float, nargs="+", default=P_GRID)
    args = ap.parse_args()

    if args.stage in ("recal", "zne", "all"):
        print(f"[recal/zne] {args.seeds} seeds, p = {args.p_grid}")
        run_recal_zne(args.seeds, args.p_grid,
                      do_recal=args.stage in ("recal", "all"),
                      do_zne=args.stage in ("zne", "all"))
    if args.stage in ("design", "all"):
        print(f"[design] depths {DEPTHS} x {ENTANGLEMENTS}, {args.seeds} seeds")
        run_design(args.seeds, DESIGN_P)
    if args.stage in ("report", "all"):
        make_report()


if __name__ == "__main__":
    main()
