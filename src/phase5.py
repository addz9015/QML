"""Phase 5 - run the trained circuit on real IBM Quantum hardware.

For a handful of held-out test images, compares:
  ideal simulator   default.qubit, exact (no shots)
  noisy simulator   default.mixed, matched depolarizing noise, exact (Phase 3's noise model)
  real IBM backend  one Estimator job with one pub (circuit) per image, real device noise +
                    finite shots

All images are submitted as separate pubs within a SINGLE Runtime job (not one job per image);
this is what actually keeps quota use low, since job start-up overhead is paid once.

`--fake` substitutes a local noise-model simulator (mimics a real backend, zero cost) for the
real backend - use it to test this script itself before spending any quantum runtime quota.

    python -m src.phase5 --stage smoke              # 1 image, cheapest possible real-hardware check
    python -m src.phase5 --stage smoke --fake       # same, zero cost (local fake backend)
    python -m src.phase5 --stage run --n 10 --shots 1024   # the real comparison, one job
    python -m src.phase5 --stage report
    python -m src.phase5 --stage zne --n 10 --shots 1024   # gate-folding ZNE, see below

Each stage that touches the real backend prints how many circuit evaluations it is about to
submit and asks for confirmation, since quantum runtime quota is limited and this spends it.
Pass --yes to skip the prompt for non-interactive runs.

`--stage zne` re-uses the fold=0 (unmitigated) hardware data already collected by `--stage run`
for the same --n (run that first) and only submits the NEW fold=1 and fold=2 circuits - all
images x both new fold levels as pubs in one job - then Richardson-extrapolates <Z> back to
zero noise per image, the same method as Phase 4's `zne-fold-rich`.
"""

import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from model import HybridModel, to_tensor
from quantum_model import make_qnode, run_qnode, run_on_backend
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import load_split, resnet_features
from feature_extractor import get_feature_extractor

RAW = os.path.join(CFG.results_dir, "phase5_raw.csv")
FIELDS = ["stage", "seed_images", "backend", "image", "true_label", "q0", "q1", "q2", "q3"]
COMPARISON_NOISE = {"type": "depolarizing", "p": 0.1}   # Phase 3/4's mid-strength condition

ZNE_RAW = os.path.join(CFG.results_dir, "phase5_zne_raw.csv")
ZNE_FIELDS = ["image", "true_label", "fold", "q0", "q1", "q2", "q3",
             "ideal_q0", "ideal_q1", "ideal_q2", "ideal_q3"]
FOLD_LAMS = (1.0, 3.0, 5.0)   # noise amplification at fold = 0, 1, 2 (matches Phase 4)


def load_model():
    saved = joblib.load(CFG.model_path)
    model = HybridModel(in_dim=saved["in_dim"], n_qubits=saved["n_qubits"],
                        n_layers=saved["n_layers"], use_circuit=saved.get("use_circuit", True),
                        entanglement=saved.get("entanglement", "ring"))
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved["scaler"], saved["classes"]


def pick_images(n):
    """n test images, balanced across classes, in a fixed order (same every run)."""
    images, labels, classes = load_split(CFG.test_dir)
    idx = np.concatenate([np.where(labels == c)[0][:n // 2 + n % 2] for c in (0, 1)])[:n]
    resnet, device = get_feature_extractor()
    feats = resnet_features(resnet, device, images[idx])
    return feats, labels[idx], classes


def get_backend(fake):
    from qiskit_ibm_runtime import QiskitRuntimeService
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        print("[fake backend] FakeSherbrooke (local noise-model simulator, no quota used)")
        return FakeSherbrooke()
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, min_num_qubits=CFG.n_qubits)
    print(f"[real backend] {backend.name}  (queue: {backend.status().pending_jobs} jobs)")
    return backend


def confirm(n_circuits, shots, fake, assume_yes):
    if fake:
        return True
    print(f"\nAbout to submit {n_circuits} circuit(s) x {shots} shots to REAL IBM hardware. "
         "This spends your account's quantum runtime quota.")
    if assume_yes:
        print("--yes given: proceeding without an interactive prompt.")
        return True
    return input("Type 'yes' to continue: ").strip().lower() == "yes"


def run(stage, n, shots, fake, assume_yes=False):
    model, scaler, classes = load_model()
    feats, labels, _ = pick_images(n)
    X = scaler.transform(feats)
    angles = model.encode(to_tensor(X)).detach().numpy()
    weights = model.qlayer.weights.detach().numpy()

    ideal = run_qnode(make_qnode(model.n_qubits, model.n_layers, None, model.entanglement),
                      angles, weights)
    noisy = run_qnode(make_qnode(model.n_qubits, model.n_layers, COMPARISON_NOISE,
                                 model.entanglement), angles, weights)

    if not confirm(len(angles), shots, fake, assume_yes):
        print("Cancelled - nothing was submitted.")
        return

    backend = get_backend(fake)
    t0 = time.time()
    hw, job = run_on_backend(backend, angles, weights, model.n_qubits, model.n_layers,
                             model.entanglement, shots=shots)
    print(f"hardware run took {time.time() - t0:.0f}s for {len(angles)} image(s) "
         f"in one job ({job.job_id() if hasattr(job, 'job_id') else 'local'})")

    rows = []
    for i in range(len(angles)):
        for tag, q in (("ideal", ideal[i]), ("noisy_sim", noisy[i]), ("hardware", hw[i])):
            rows.append({"stage": stage, "seed_images": n, "backend": tag, "image": i,
                        "true_label": int(labels[i]), "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3]})
    os.makedirs(CFG.results_dir, exist_ok=True)
    write_header = not os.path.exists(RAW)
    with open(RAW, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {RAW}")

    head_w = model.head.weight.detach().numpy()
    head_b = model.head.bias.detach().numpy()

    def acc_of(q):
        logits = q @ head_w.T + head_b
        return float((logits.argmax(1) == labels).mean())

    print(f"\naccuracy on these {n} image(s): "
         f"ideal={acc_of(ideal):.3f}  noisy_sim(p={COMPARISON_NOISE['p']})={acc_of(noisy):.3f}  "
         f"hardware={acc_of(hw):.3f}")
    print("mean |hardware - ideal| per qubit:", np.round(np.abs(hw - ideal).mean(0), 3))


def run_zne(n, shots, fake, assume_yes):
    from phase4 import extrapolate_to_zero
    import pandas as pd

    if not os.path.exists(RAW):
        raise SystemExit("Run --stage run first - ZNE reuses its fold=0 hardware data.")
    df = pd.read_csv(RAW)
    hw0 = df[(df.backend == "hardware") & (df.seed_images == n)].sort_values("image")
    if len(hw0) != n:
        raise SystemExit(f"Need {n} existing fold=0 hardware rows (seed_images={n}); "
                         f"found {len(hw0)}. Run: python -m src.phase5 --stage run --n {n}")
    fold0 = hw0[["q0", "q1", "q2", "q3"]].to_numpy()
    labels = hw0["true_label"].to_numpy()

    model, scaler, classes = load_model()
    feats, labels2, _ = pick_images(n)
    if not (labels2 == labels).all():
        raise SystemExit("data/test changed since --stage run; re-run --stage run first.")
    X = scaler.transform(feats)
    angles = model.encode(to_tensor(X)).detach().numpy()
    weights = model.qlayer.weights.detach().numpy()
    ideal = run_qnode(make_qnode(model.n_qubits, model.n_layers, None, model.entanglement),
                      angles, weights)

    # fold=0 is already measured (reused above); only fold 1 and 2 need new hardware time.
    angles_batch = np.concatenate([angles, angles])
    fold_batch = np.repeat([1, 2], n)

    if not confirm(len(angles_batch), shots, fake, assume_yes):
        print("Cancelled - nothing was submitted.")
        return

    backend = get_backend(fake)
    t0 = time.time()
    out, job = run_on_backend(backend, angles_batch, weights, model.n_qubits, model.n_layers,
                              model.entanglement, fold=fold_batch, shots=shots)
    print(f"ZNE hardware run took {time.time() - t0:.0f}s for {len(angles_batch)} circuits "
         f"(fold 1 and 2, {n} images each) in one job "
         f"({job.job_id() if hasattr(job, 'job_id') else 'local'})")
    fold1, fold2 = out[:n], out[n:]

    zne = np.array([extrapolate_to_zero(FOLD_LAMS, [fold0[i], fold1[i], fold2[i]], order=2)
                    for i in range(n)])

    rows = []
    for i in range(n):
        for f, q in ((0, fold0[i]), (1, fold1[i]), (2, fold2[i])):
            rows.append({"image": i, "true_label": int(labels[i]), "fold": f,
                        "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3],
                        "ideal_q0": ideal[i, 0], "ideal_q1": ideal[i, 1],
                        "ideal_q2": ideal[i, 2], "ideal_q3": ideal[i, 3]})
    os.makedirs(CFG.results_dir, exist_ok=True)
    with open(ZNE_RAW, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ZNE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {ZNE_RAW}")

    head_w = model.head.weight.detach().numpy()
    head_b = model.head.bias.detach().numpy()

    def acc_of(q):
        return float(((q @ head_w.T + head_b).argmax(1) == labels).mean())

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"\nRMSE vs ideal   fold=0(raw)={rmse(fold0, ideal):.4f}  fold=1={rmse(fold1, ideal):.4f}  "
         f"fold=2={rmse(fold2, ideal):.4f}  ZNE(Richardson)={rmse(zne, ideal):.4f}")
    print(f"accuracy        fold=0(raw)={acc_of(fold0):.3f}  fold=1={acc_of(fold1):.3f}  "
         f"fold=2={acc_of(fold2):.3f}  ZNE(Richardson)={acc_of(zne):.3f}")

    labels_pts = ["fold=0\n(raw hw)", "fold=1", "fold=2", "ZNE\n(Richardson)"]
    rmses = [rmse(fold0, ideal), rmse(fold1, ideal), rmse(fold2, ideal), rmse(zne, ideal)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels_pts, rmses, color=["#888888", "#f4a261", "#e76f51", "#2a9d8f"])
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("RMSE of <Z> vs ideal simulator")
    ax.set_title(f"Phase 5: gate-folding ZNE on real hardware ({n} images)")
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, "phase5_zne.png"), dpi=140)
    print("\nplot -> results/phase5_zne.png")


def make_report():
    import pandas as pd
    if not os.path.exists(RAW):
        raise SystemExit("No Phase 5 results yet - run --stage run first.")
    df = pd.read_csv(RAW)
    piv = df.pivot_table(index=["image"], columns="backend", values=["q0", "q1", "q2", "q3"])
    print(piv)

    fig, ax = plt.subplots(figsize=(6, 4))
    for backend, marker in (("ideal", "o"), ("noisy_sim", "s"), ("hardware", "^")):
        sub = df[df.backend == backend]
        z = sub[["q0", "q1", "q2", "q3"]].to_numpy()
        ax.plot(np.repeat(sub["image"].to_numpy(), 4), z.ravel(), marker, alpha=0.6, label=backend)
    ax.set_xlabel("test image")
    ax.set_ylabel("<Z_i>")
    ax.set_title("Phase 5: ideal vs noisy-simulator vs real hardware")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CFG.results_dir, "phase5_comparison.png"), dpi=140)
    print("\nplot -> results/phase5_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "run", "zne", "report"], default="smoke")
    ap.add_argument("--n", type=int, default=10, help="number of test images (run stage)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--fake", action="store_true",
                    help="use a local noise-model backend instead of real hardware (no quota used)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation (for non-interactive / scripted runs)")
    args = ap.parse_args()

    if args.stage == "smoke":
        run("smoke", n=1, shots=min(args.shots, 512), fake=args.fake, assume_yes=args.yes)
    elif args.stage == "run":
        run("run", n=args.n, shots=args.shots, fake=args.fake, assume_yes=args.yes)
    elif args.stage == "zne":
        run_zne(n=args.n, shots=args.shots, fake=args.fake, assume_yes=args.yes)
    else:
        make_report()


if __name__ == "__main__":
    main()
