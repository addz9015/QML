import os
import sys
import argparse

# jaxlib DLL load order on Windows: pennylane (via model/quantum_model) before scipy.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from model import HybridModel, fit, to_tensor
from tracking import log_run

import torch
import joblib
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocessing import get_transform
from feature_extractor import get_feature_extractor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 16


def list_class_images(class_dir):
    out = []
    for dirpath, _, names in os.walk(class_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def load_split(root):
    """Load an image folder laid out as root/<class>/*.jpg -> (tensor, labels, classes)."""
    transform = get_transform()
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    images, labels = [], []
    for label, name in enumerate(classes):
        for fp in list_class_images(os.path.join(root, name)):
            try:
                images.append(transform(Image.open(fp).convert("RGB")))
                labels.append(label)
            except Exception as exc:
                print("  skipped", fp, "->", exc)
    if not images:
        raise SystemExit(f"No images under {root} (run: python -m src.build_dataset)")
    return torch.stack(images), np.array(labels), classes


def resnet_features(model, device, images):
    feats = []
    for s in range(0, len(images), BATCH_SIZE):
        with torch.no_grad():
            feats.append(model(images[s:s + BATCH_SIZE].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def get_train_features(refresh=False):
    if not refresh and os.path.exists(CFG.feature_cache):
        d = np.load(CFG.feature_cache, allow_pickle=True)
        print(f"Loaded cached ResNet features {d['features'].shape}")
        return d["features"], d["labels"], list(d["classes"])

    images, labels, classes = load_split(CFG.train_dir)
    print("Train images:", len(images), "| per class:",
          {c: int((labels == i).sum()) for i, c in enumerate(classes)})
    model, device = get_feature_extractor()
    features = resnet_features(model, device, images)
    os.makedirs(os.path.dirname(CFG.feature_cache), exist_ok=True)
    np.savez(CFG.feature_cache, features=features, labels=labels,
             classes=np.array(classes))
    return features, labels, classes


def get_test_features(refresh=False):
    """Held-out test features (raw ResNet output), cached like the train features."""
    if not refresh and os.path.exists(CFG.test_feature_cache):
        d = np.load(CFG.test_feature_cache, allow_pickle=True)
        return d["features"], d["labels"], list(d["classes"])

    images, labels, classes = load_split(CFG.test_dir)
    model, device = get_feature_extractor()
    features = resnet_features(model, device, images)
    os.makedirs(os.path.dirname(CFG.test_feature_cache), exist_ok=True)
    np.savez(CFG.test_feature_cache, features=features, labels=labels,
             classes=np.array(classes))
    return features, labels, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=["trained", "fixed", "none"], default="trained",
                    help="trained=Phase 2 VQC, fixed=frozen random circuit (Phase 1), "
                         "none=classical baseline (no circuit)")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    features, labels, classes = get_train_features(refresh=args.refresh)

    scaler = StandardScaler().fit(features)
    X = scaler.transform(features)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=CFG.seed
    )

    model = HybridModel(in_dim=X.shape[1], n_qubits=CFG.n_qubits,
                        n_layers=CFG.n_layers,
                        freeze_circuit=(args.circuit == "fixed"),
                        use_circuit=(args.circuit != "none"),
                        entanglement=CFG.entanglement)

    print(f"\nTraining hybrid model (circuit={args.circuit}, "
          f"{CFG.n_qubits} qubits, {CFG.n_layers} re-uploading layers)...")
    fit(model,
        to_tensor(X_tr), to_tensor(y_tr, torch.long),
        to_tensor(X_val), to_tensor(y_val, torch.long),
        epochs=args.epochs, lr=args.lr, weight_decay=CFG.weight_decay, seed=CFG.seed)

    # ---- report on the internal val split ----
    val_proba = model.predict_proba(to_tensor(X_val))
    val_acc = accuracy_score(y_val, val_proba.argmax(1))
    val_auc = roc_auc_score(y_val, val_proba[:, 1])
    print(f"\ncircuit mode  : {args.circuit}")
    print(f"val accuracy  : {val_acc:.3f}")
    print(f"val ROC-AUC   : {val_auc:.3f}")
    log_run(phase="1-2", circuit=args.circuit, eval="val",
            n_qubits=CFG.n_qubits, n_layers=CFG.n_layers, n_train=len(y_tr),
            accuracy=round(float(val_acc), 4), roc_auc=round(float(val_auc), 4))

    # ---- report on the held-out test set if present ----
    test_root = CFG.test_dir
    if any(os.path.isdir(os.path.join(test_root, d)) for d in os.listdir(test_root)):
        t_images, t_labels, _ = load_split(test_root)
        rmodel, device = get_feature_extractor()
        t_feat = scaler.transform(resnet_features(rmodel, device, t_images))
        t_proba = model.predict_proba(to_tensor(t_feat))
        t_acc = accuracy_score(t_labels, t_proba.argmax(1))
        t_auc = roc_auc_score(t_labels, t_proba[:, 1])
        print(f"\nheld-out test ({len(t_labels)} imgs)")
        print(f"  accuracy : {t_acc:.3f}")
        print(f"  ROC-AUC  : {t_auc:.3f}")
        log_run(phase="1-2", circuit=args.circuit, eval="test",
                n_qubits=CFG.n_qubits, n_layers=CFG.n_layers, n_train=len(labels),
                accuracy=round(float(t_acc), 4), roc_auc=round(float(t_auc), 4))

    # ---- refit on all training data for the saved model ----
    fit(model, to_tensor(X), to_tensor(labels, torch.long),
        epochs=args.epochs, lr=args.lr, weight_decay=CFG.weight_decay,
        seed=CFG.seed, verbose=False)

    os.makedirs(os.path.dirname(CFG.model_path), exist_ok=True)
    joblib.dump(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "classes": classes,
            "in_dim": X.shape[1],
            "n_qubits": CFG.n_qubits,
            "n_layers": CFG.n_layers,
            "entanglement": CFG.entanglement,
            "circuit_mode": args.circuit,
            "use_circuit": args.circuit != "none",
        },
        CFG.model_path,
    )
    print("\nModel saved to", CFG.model_path)


if __name__ == "__main__":
    main()
