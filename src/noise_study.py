"""Phase 3 - how does quantum noise affect the hybrid classifier?

Noise model: after every circuit layer, one noise channel is applied to every
qubit (bit-flip, phase-flip or depolarizing, strength p), simulated exactly on
PennyLane's density-matrix device. All numbers are on the held-out test set
(data/test, 200 images) and averaged over CFG.n_seeds independent trainings.

  inference : models trained on the IDEAL circuit are run with noise switched on.
              ("what if a noise-free model is moved onto a noisy machine?")
  aware     : models trained WITH the noise switched on (noise-aware training).
              ("can the model learn to work around the noise?")

    python -m src.noise_study --stage inference   # minutes (trains + caches ideal models)
    python -m src.noise_study --stage aware       # ~1-2 h, resumable
    python -m src.noise_study --stage report      # tables + plots from saved results

Outputs: results/phase3_raw.csv, phase3_summary.md, phase3_*.png, metrics.csv rows.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (CFG, NOISE_TYPES, Data, seed_split, get_ideal, train_model, qfeatures,
                         head_proba, score, row, noise_spec, load_rows, upsert_rows, select,
                         by_seed, mean_sd, paired)
from tracking import log_run

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = os.path.join(CFG.results_dir, "phase3_raw.csv")
P_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
AWARE_P = [0.1, 0.15, 0.2, 0.3]
AWARE_WD = 0.01               # weight decay for noise-aware training, chosen on validation accuracy
CONTROL = "ideal-wd-matched"  # ideal-trained models that use AWARE_WD (control experiment)
COLORS = {"bit_flip": "#d95f02", "phase_flip": "#7570b3", "depolarizing": "#1b9e77"}
LABEL = {"bit_flip": "bit-flip", "phase_flip": "phase-flip", "depolarizing": "depolarizing"}


# ------------------------------------------------------------------------ experiments
def run_inference(n_seeds, p_grid):
    """Ideal-trained models evaluated under noise (no retraining)."""
    data = Data()
    rows = []
    for seed in range(n_seeds):
        t0 = time.time()
        model, split = get_ideal(data, seed)
        for kind in NOISE_TYPES:
            for p in p_grid:
                q = qfeatures(model, split["Xte"], noise_spec(kind, p))
                rows.append(row("phase3", "ideal-trained", seed, kind, p, epochs=CFG.epochs,
                                **score(head_proba(model, q), split["yte"], q)))
        clean = select(rows, seed=seed, p=0.0)[0]
        print(f"  seed {seed}: noise-free test acc={clean['accuracy']:.3f} "
              f"auc={clean['roc_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    upsert_rows(RAW, rows)


def run_aware(n_seeds, aware_p, epochs, force, weight_decay=None):
    """Train with the noise on; evaluate on the test set under the same noise."""
    data = Data()
    for seed in range(n_seeds):
        split = seed_split(data, seed)
        for kind in NOISE_TYPES:
            for p in aware_p:
                have = select(load_rows(RAW), stage="phase3", method="aware", noise_type=kind,
                              p=round(p, 6), seed=seed)
                if have and not force:
                    print(f"  skip {kind} p={p} seed={seed} (done)")
                    continue
                t0 = time.time()
                noise = noise_spec(kind, p)
                model = train_model(split, seed, noise=noise, epochs=epochs,
                                    weight_decay=weight_decay)
                q = qfeatures(model, split["Xte"], noise)
                s = score(head_proba(model, q), split["yte"], q)
                upsert_rows(RAW, [row("phase3", "aware", seed, kind, p, epochs=epochs, **s)])
                print(f"  {kind:13s} p={p:<5} seed={seed}: acc={s['accuracy']:.3f} "
                      f"auc={s['roc_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)


def run_control(n_seeds, p_grid, weight_decay):
    """Control for a confound: noise-aware models use a smaller weight decay than the
    ideal-trained ones. Train IDEAL models (no noise while training) with that same weight
    decay and run them under noise. If they fail too, the gain comes from noise-aware training."""
    data = Data()
    rows = []
    for seed in range(n_seeds):
        split = seed_split(data, seed)
        model = train_model(split, seed, weight_decay=weight_decay)
        for kind in NOISE_TYPES:
            for p in p_grid:
                q = qfeatures(model, split["Xte"], noise_spec(kind, p))
                rows.append(row("phase3", CONTROL, seed, kind, p, epochs=CFG.epochs,
                                **score(head_proba(model, q), split["yte"], q)))
        print(f"  seed {seed}: control model trained", flush=True)
    upsert_rows(RAW, rows)


# ---------------------------------------------------------------------------- report
def imbalance(r):
    """0 = balanced predictions, 0.5 = every image gets the same label (per seed, so
    seeds that collapse towards opposite classes cannot cancel out)."""
    return abs(r["frac_pred_class1"] - 0.5)


def curve(rows, method, kind, field):
    """(p values, mean, sd) over seeds for one method / noise type."""
    ps = sorted({r["p"] for r in select(rows, method=method, noise_type=kind)})
    m, s = [], []
    for p in ps:
        a, b = mean_sd(r[field] for r in select(rows, method=method, noise_type=kind, p=p))
        m.append(a)
        s.append(b)
    return np.array(ps), np.array(m), np.array(s)


def make_report():
    rows = load_rows(RAW)
    if not rows:
        raise SystemExit("No Phase 3 results yet: run --stage inference first.")
    os.makedirs(CFG.results_dir, exist_ok=True)
    n_seeds = len({r["seed"] for r in rows})

    for field, label, fname in (("accuracy", "test accuracy", "phase3_accuracy.png"),
                                ("roc_auc", "test ROC-AUC", "phase3_auc.png")):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
        for ax, kind in zip(axes, NOISE_TYPES):
            x, m, s = curve(rows, "ideal-trained", kind, field)
            ax.plot(x, m, "o-", color=COLORS[kind], label="ideal-trained, run noisy")
            ax.fill_between(x, m - s, m + s, color=COLORS[kind], alpha=0.2)
            xc, mc, _ = curve(rows, CONTROL, kind, field)
            if len(xc):
                ax.plot(xc, mc, "x:", color="grey", label="ideal-trained, same weight decay")
            xa, ma, sa = curve(rows, "aware", kind, field)
            if len(xa):
                ax.errorbar(xa, ma, yerr=sa, fmt="s--", color="black", capsize=3,
                            label="noise-aware training")
            ax.axhline(0.5, color="grey", ls=":", lw=1)
            ax.set_title(LABEL[kind])
            ax.set_xlabel("noise probability p")
            ax.set_ylim(0.4, 1.02)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(label)
        axes[0].legend(loc="lower left", fontsize=8)
        fig.suptitle(f"Phase 3: {label} vs noise (mean ± sd over {n_seeds} seeds; dotted = chance)")
        fig.tight_layout()
        fig.savefig(os.path.join(CFG.results_dir, fname), dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for kind in NOISE_TYPES:
        ps = sorted({r["p"] for r in select(rows, method="ideal-trained", noise_type=kind)})
        for ax, field in zip(axes, ("mean_abs_z", "imbalance")):
            vals = []
            for p in ps:
                sel = select(rows, method="ideal-trained", noise_type=kind, p=p)
                vals.append(mean_sd(imbalance(r) if field == "imbalance" else r[field] for r in sel))
            m, s = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            ax.plot(ps, m, "o-", color=COLORS[kind], label=LABEL[kind])
            ax.fill_between(ps, m - s, m + s, color=COLORS[kind], alpha=0.15)
    axes[0].set_ylabel("mean |<Z>| (signal strength)")
    axes[1].set_ylabel("prediction imbalance  |share 'dog' - 0.5|")
    for ax in axes:
        ax.set_xlabel("noise probability p")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Phase 3: noise shrinks the quantum signal and shifts the decision threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, "phase3_signal.png"), dpi=140)
    plt.close(fig)

    lines = [f"Held-out test set: {2 * CFG.test_per_class} images; mean ± sd over {n_seeds} seeds. "
             f"Ideal-trained models use weight decay {CFG.weight_decay:g}, noise-aware ones {AWARE_WD:g} "
             "(each chosen on validation accuracy); 'ideal, wd matched' is the control: ideal-trained "
             f"with weight decay {AWARE_WD:g}. 'aware − ideal' is the paired difference against the "
             "ideal-trained column (mean ± 95% CI); 'better in' counts seeds where noise-aware "
             "training beat it. 'imbalance' = |share predicted dog − 0.5| (0.5 = one class only).",
             "",
             "| noise | p | ideal-trained acc | AUC | imbalance | mean abs Z | ideal, wd matched | noise-aware acc | aware − ideal | better in |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for kind in NOISE_TYPES:
        for p in sorted({r["p"] for r in select(rows, method="ideal-trained", noise_type=kind)}):
            ide = select(rows, method="ideal-trained", noise_type=kind, p=p)
            a, asd = mean_sd(r["accuracy"] for r in ide)
            auc, _ = mean_sd(r["roc_auc"] for r in ide)
            imb, _ = mean_sd(imbalance(r) for r in ide)
            z, _ = mean_sd(r["mean_abs_z"] for r in ide)
            ctl = select(rows, method=CONTROL, noise_type=kind, p=p)
            ctl_txt = "%.3f ± %.3f" % mean_sd(r["accuracy"] for r in ctl) if ctl else "-"
            aw = select(rows, method="aware", noise_type=kind, p=p)
            if aw:
                am, asd2 = mean_sd(r["accuracy"] for r in aw)
                d, ci, wins, n = paired(by_seed(ide, "accuracy"), by_seed(aw, "accuracy"))
                aw_txt, d_txt, w_txt = f"{am:.3f} ± {asd2:.3f}", f"{d:+.3f} ± {ci:.3f}", f"{wins}/{n}"
            else:
                aw_txt = d_txt = w_txt = "-"
            lines.append(f"| {LABEL[kind]} | {p:g} | {a:.3f} ± {asd:.3f} | {auc:.3f} | {imb:.2f} "
                         f"| {z:.3f} | {ctl_txt} | {aw_txt} | {d_txt} | {w_txt} |")
    text = "\n".join(lines)
    with open(os.path.join(CFG.results_dir, "phase3_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)

    # aggregated rows in the shared metrics log
    for method in ("ideal-trained", "aware"):
        for kind in NOISE_TYPES:
            for p in sorted({r["p"] for r in select(rows, method=method, noise_type=kind)}):
                sel = select(rows, method=method, noise_type=kind, p=p)
                a, sd = mean_sd(r["accuracy"] for r in sel)
                auc, _ = mean_sd(r["roc_auc"] for r in sel)
                log_run(phase="3", circuit="trained", noise_type=kind, noise_p=p,
                        n_qubits=CFG.n_qubits, n_layers=CFG.n_layers,
                        eval=f"test/{method}", accuracy=round(a, 4),
                        accuracy_std=round(sd, 4), roc_auc=round(auc, 4))


# ------------------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inference", "aware", "control", "report", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=CFG.n_seeds)
    ap.add_argument("--p-grid", type=float, nargs="+", default=P_GRID)
    ap.add_argument("--aware-p", type=float, nargs="+", default=AWARE_P)
    ap.add_argument("--aware-epochs", type=int, default=CFG.epochs)
    ap.add_argument("--aware-wd", type=float, default=AWARE_WD,
                    help="weight decay for noise-aware training and for the control")
    ap.add_argument("--force", action="store_true", help="redo finished aware runs")
    args = ap.parse_args()

    if args.stage in ("inference", "all"):
        print(f"[inference] {args.seeds} seeds x {len(NOISE_TYPES)} noise types x {len(args.p_grid)} strengths")
        run_inference(args.seeds, args.p_grid)
    if args.stage in ("aware", "all"):
        print(f"[aware] p={args.aware_p}, {args.seeds} seeds, {args.aware_epochs} epochs")
        run_aware(args.seeds, args.aware_p, args.aware_epochs, args.force, args.aware_wd)
    if args.stage in ("control", "all"):
        print(f"[control] ideal-trained models with weight decay {args.aware_wd:g}")
        run_control(args.seeds, [0.0] + list(args.aware_p), args.aware_wd)
    if args.stage in ("report", "all"):
        make_report()


if __name__ == "__main__":
    main()
