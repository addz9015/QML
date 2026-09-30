import os
import sys
import glob

# jaxlib DLL load order on Windows: pennylane (via model) before scipy/sklearn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from model import HybridModel, to_tensor
from quantum_model import layer_trace

import torch
import joblib
import numpy as np

from PIL import Image

from preprocessing import get_transform
from feature_extractor import get_feature_extractor

_PIPELINE = None


def load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not os.path.exists(CFG.model_path):
            raise FileNotFoundError(
                f"{CFG.model_path} not found. Train first: python -m src.train"
            )
        saved = joblib.load(CFG.model_path)

        model = HybridModel(in_dim=saved["in_dim"], n_qubits=saved["n_qubits"],
                            n_layers=saved["n_layers"],
                            use_circuit=saved.get("use_circuit", True),
                            entanglement=saved.get("entanglement", "ring"))
        model.load_state_dict(saved["model_state"])
        model.eval()

        resnet, device = get_feature_extractor()
        _PIPELINE = {
            "model": model,
            "scaler": saved["scaler"],
            "classes": saved["classes"],
            "circuit_mode": saved.get("circuit_mode", "trained"),
            "transform": get_transform(),
            "resnet": resnet,
            "device": device,
        }
    return _PIPELINE


def _model_input(image_path):
    """Image file -> standardised 512-d ResNet feature tensor (1, 512), as used in training."""
    p = load_pipeline()
    image = Image.open(image_path).convert("RGB")
    tensor = p["transform"](image).unsqueeze(0).to(p["device"])
    with torch.no_grad():
        feat = p["resnet"](tensor).cpu().numpy()
    return to_tensor(p["scaler"].transform(feat))


def predict_image(image_path):
    p = load_pipeline()
    x = _model_input(image_path)

    probabilities = p["model"].predict_proba(x)[0]
    q_features = p["model"].quantum_features(x)[0]
    prediction = int(probabilities.argmax())

    classes = p["classes"]
    return {
        "class": classes[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": {c: float(v) for c, v in zip(classes, probabilities)},
        "quantum_features": np.asarray(q_features, dtype=float),
    }


def explain_image(image_path):
    """What the quantum circuit does for one image (None for the classical baseline).

    angles : the n_qubits rotation angles the trained projection feeds into the circuit
    trace  : <Z_i> after each layer, shape (n_layers, n_qubits); the last row is exactly
             the quantum feature vector the readout sees
    """
    p = load_pipeline()
    model = p["model"]
    if not model.use_circuit:
        return None
    x = _model_input(image_path)
    with torch.no_grad():
        angles = model.encode(x)
        trace = layer_trace(angles, model.qlayer.weights.detach(), model.n_qubits,
                            model.n_layers, model.entanglement)
    return {"angles": angles[0].numpy().astype(float), "trace": trace[:, 0, :]}


if __name__ == "__main__":
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(glob.glob(os.path.join(CFG.test_dir, "*.jpg")))
    if not targets:
        raise SystemExit("No images. Pass paths, or put .jpg files in data/test/")

    for path in targets:
        r = predict_image(path)
        print(f"\n{os.path.basename(path)}")
        print(f"  prediction : {r['class']}")
        print(f"  confidence : {r['confidence'] * 100:.2f}%")
        print(f"  probs      : {r['probabilities']}")
        print(f"  q-features : {np.round(r['quantum_features'], 4)}")
