"""Integrity checks for the Phase 3 / Phase 4 result files.

    python -m src.check_results

Fails (exit code 1) if any of these does not hold:
  * every accuracy / AUC / share is a valid probability (0..1) - no NaNs
  * every experiment cell has exactly CFG.n_seeds independent runs
  * every expected cell exists (all noise types x strengths x methods x designs)
  * the noise-free rows are identical for the three noise types
  * Phase 4 "none" (no mitigation) equals Phase 3 "ideal-trained" run for run
  * mitigation never changes ROC-AUC of a re-thresholded readout (threshold method)
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import CFG, NOISE_TYPES, load_rows, select
from quantum_model import ENTANGLEMENTS
import noise_study
import phase4

failures = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def cells(rows, *fields):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[f] for f in fields), []).append(r)
    return out


def main():
    r3 = load_rows(noise_study.RAW)
    r4 = load_rows(phase4.RAW)
    check(bool(r3), f"Phase 3 results present ({len(r3)} rows)")
    check(bool(r4), f"Phase 4 results present ({len(r4)} rows)")
    n = CFG.n_seeds

    # --- value sanity -------------------------------------------------------------------
    for name, rows in (("phase3", r3), ("phase4", r4)):
        bad = [r for r in rows
               if any(not (0.0 <= r[f] <= 1.0) or math.isnan(r[f])
                      for f in ("accuracy", "roc_auc", "frac_pred_class1"))]
        check(not bad, f"{name}: accuracy / AUC / prediction share all valid probabilities in [0, 1]")

    # --- each cell has n seeds ----------------------------------------------------------
    for name, rows in (("phase3", r3), ("phase4", r4)):
        groups = cells(rows, "stage", "method", "depth", "entanglement", "noise_type", "p")
        short = {k: len(v) for k, v in groups.items() if len(v) != n}
        check(not short, f"{name}: every cell has exactly {n} seeds"
                         + (f" (off: {list(short.items())[:3]})" if short else ""))
        dup = [k for k, v in groups.items() if len({r['seed'] for r in v}) != len(v)]
        check(not dup, f"{name}: no duplicated seed within a cell")

    # --- expected coverage --------------------------------------------------------------
    def have(rows, **kw):
        return len(select(rows, **kw)) == n

    miss = [(t, p) for t in NOISE_TYPES for p in noise_study.P_GRID
            if not have(r3, method="ideal-trained", noise_type=t, p=p)]
    check(not miss, "phase3: ideal-trained results for every noise type x strength" + (f" missing {miss[:3]}" if miss else ""))
    miss = [(t, p) for t in NOISE_TYPES for p in noise_study.AWARE_P
            if not have(r3, method="aware", noise_type=t, p=p)]
    check(not miss, "phase3: noise-aware results for every noise type x strength" + (f" missing {miss[:3]}" if miss else ""))

    methods = ["none", "threshold", "head-refit", "zne-scale-lin", "zne-scale-rich",
               "zne-fold-lin", "zne-fold-rich"]
    main4 = [r for r in r4 if r["stage"] == "phase4"]
    miss = [(m, t, p) for m in methods for t in NOISE_TYPES for p in phase4.P_GRID
            if not have(main4, method=m, noise_type=t, p=p)]
    check(not miss, "phase4: every mitigation x noise type x strength" + (f" missing {miss[:3]}" if miss else ""))

    design = [r for r in r4 if r["stage"] == "design"]
    miss = [(d, e, t, p) for d in phase4.DEPTHS for e in ENTANGLEMENTS for t in NOISE_TYPES
            for p in phase4.DESIGN_P if not have(design, depth=d, entanglement=e, method="none",
                                                 noise_type=t, p=p)]
    check(not miss, "phase4: every circuit design x noise type x strength" + (f" missing {miss[:3]}" if miss else ""))

    # --- internal consistency -----------------------------------------------------------
    clean = {t: sorted((r["seed"], r["accuracy"]) for r in select(r3, method="ideal-trained", noise_type=t, p=0.0))
             for t in NOISE_TYPES}
    check(len({tuple(v) for v in clean.values()}) == 1, "phase3: noise-free rows identical for all noise types")

    worst = 0.0
    for r in select(main4, method="none"):
        m = select(r3, method="ideal-trained", noise_type=r["noise_type"], p=r["p"], seed=r["seed"])
        if m:
            worst = max(worst, abs(m[0]["accuracy"] - r["accuracy"]))
    check(worst < 1e-9, f"phase4 'none' == phase3 'ideal-trained' run for run (max |diff| = {worst:.2g})")

    worst = 0.0
    for r in select(main4, method="threshold"):
        m = select(main4, method="none", noise_type=r["noise_type"], p=r["p"], seed=r["seed"])
        if m:
            worst = max(worst, abs(m[0]["roc_auc"] - r["roc_auc"]))
    check(worst < 1e-9, f"re-thresholding leaves ROC-AUC unchanged (max |diff| = {worst:.2g})")

    dm = {(r["depth"], r["entanglement"], r["noise_type"], r["seed"]): r["accuracy"]
          for r in select(design, method="none", p=0.0)}
    main_design = [dm[(CFG.n_layers, CFG.entanglement, t, s)] for t in NOISE_TYPES for s in range(n)
                   if (CFG.n_layers, CFG.entanglement, t, s) in dm]
    base = [r["accuracy"] for t in NOISE_TYPES for r in select(r3, method="ideal-trained", noise_type=t, p=0.0)]
    check(sorted(main_design) == sorted(base),
          "design sweep: the main design (3 layers, ring) reuses exactly the Phase 3 models")

    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
