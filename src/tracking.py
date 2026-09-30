"""Shared results log: results/metrics.csv (one row per phase / method / condition).

Logging is idempotent: a row with the same key (phase, circuit, noise type, noise p, eval)
replaces the earlier one, so re-running an experiment updates it instead of duplicating it.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from config import CFG

METRICS_CSV = os.path.join(CFG.results_dir, "metrics.csv")

FIELDS = [
    "timestamp", "phase", "circuit", "noise_type", "noise_p",
    "n_qubits", "n_layers", "n_train", "eval",
    "accuracy", "accuracy_std", "roc_auc",
]
KEY = ("phase", "circuit", "noise_type", "noise_p", "eval")


def _key(row):
    return tuple(str(row.get(k, "")) for k in KEY)


def log_run(**row) -> None:
    os.makedirs(CFG.results_dir, exist_ok=True)
    record = {f: "" for f in FIELDS}
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    record.update({k: v for k, v in row.items() if k in FIELDS})

    existing = []
    if os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, newline="") as fh:
            existing = [r for r in csv.DictReader(fh) if _key(r) != _key(record)]

    with open(METRICS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerow(record)
