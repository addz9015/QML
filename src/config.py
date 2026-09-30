"""Central hyperparameters. Import `CFG` everywhere instead of hard-coding."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # data
    train_dir: str = os.path.join(BASE_DIR, "data", "train")
    test_dir: str = os.path.join(BASE_DIR, "data", "test")
    raw_dir: str = os.path.join(BASE_DIR, "data", "raw_full")
    per_class: int = 150          # images sampled per class for train
    test_per_class: int = 100     # held-out images per class (never used for training)
    image_size: int = 224

    # artefacts
    model_path: str = os.path.join(BASE_DIR, "models", "qml_model.pkl")
    feature_cache: str = os.path.join(BASE_DIR, "models", "resnet_features.npz")
    test_feature_cache: str = os.path.join(BASE_DIR, "models", "resnet_test_features.npz")
    cache_dir: str = os.path.join(BASE_DIR, "models", "cache")   # per-seed ideal models
    results_dir: str = os.path.join(BASE_DIR, "results")

    # quantum circuit
    n_qubits: int = 4
    n_layers: int = 3             # each layer re-uploads the data (data re-uploading)
    entanglement: str = "ring"    # ring | linear | none

    # hybrid model training
    epochs: int = 60              # loss and val accuracy plateau by ~50 epochs (lr 0.05)
    lr: float = 0.05
    # L2 weight decay, chosen on VALIDATION accuracy (seeds 0-4, 60 epochs) from
    # {1e-4, 1e-3, 1e-2, 1e-1}: mean val acc 0.917 / 0.943 / 0.983 / 0.987. With 1e-4 the
    # 512->4 projection overfits and some seeds fail (test acc 0.57); >= 1e-2 is stable.
    weight_decay: float = 0.1
    seed: int = 42

    # evaluation / experiments
    cv_folds: int = 5
    n_seeds: int = 5              # independent training runs per experiment cell

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()
