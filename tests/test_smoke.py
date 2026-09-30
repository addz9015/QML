"""Fast checks - no training run and no dataset needed.

Run:  python -m pytest -q      (or)      python tests/test_smoke.py
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import numpy as np                                   # noqa: E402  (pennylane must load first)
import torch                                         # noqa: E402

from model import HybridModel, fit, to_tensor        # noqa: E402
from quantum_model import (make_qnode, run_qnode, layer_trace, draw_circuit,   # noqa: E402
                           N_QUBITS, N_LAYERS)


def _weights(n_layers=N_LAYERS, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.normal(0, 1.0, (n_layers, N_QUBITS, 2)), dtype=torch.float32)


def _angles(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.uniform(-2, 2, (n, N_QUBITS)), dtype=torch.float32)


# ------------------------------------------------------------------------ circuit
def test_circuit_output_shape_and_range():
    out = run_qnode(make_qnode(), _angles(8), _weights())
    assert out.shape == (8, N_QUBITS)
    assert np.all(np.abs(out) <= 1.0 + 1e-5)


def test_noisy_qnode_builds_and_shrinks_signal():
    ideal = np.abs(run_qnode(make_qnode(), _angles(), _weights())).mean()
    mild = np.abs(run_qnode(make_qnode(noise={"type": "depolarizing", "p": 0.02}),
                            _angles(), _weights())).mean()
    strong = np.abs(run_qnode(make_qnode(noise={"type": "depolarizing", "p": 0.2}),
                              _angles(), _weights())).mean()
    assert strong < mild < ideal


def test_folding_leaves_ideal_circuit_unchanged():
    a, w = _angles(), _weights()
    base = run_qnode(make_qnode(), a, w)
    for fold in (1, 2):
        assert np.allclose(run_qnode(make_qnode(fold=fold), a, w), base, atol=1e-5)


def test_folding_amplifies_noise():
    a, w = _angles(), _weights()
    noise = {"type": "bit_flip", "p": 0.03}
    z = [np.abs(run_qnode(make_qnode(noise=noise, fold=k), a, w)).mean() for k in (0, 1, 2)]
    assert z[2] < z[1] < z[0]


def test_entanglement_options():
    a, w = _angles(), _weights()
    outs = {e: run_qnode(make_qnode(entanglement=e), a, w) for e in ("ring", "linear", "none")}
    assert not np.allclose(outs["ring"], outs["none"])
    assert not np.allclose(outs["ring"], outs["linear"])
    try:
        make_qnode(entanglement="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown entanglement must raise")


def test_circuit_drawing():
    text = draw_circuit()
    assert "RY" in text and "RZ" in text and "<Z>" in text


# -------------------------------------------------------------------------- model
def test_hybrid_model_forward_and_probs():
    m = HybridModel(in_dim=512)
    m.eval()
    x = to_tensor(np.random.randn(5, 512))
    assert m(x).shape == (5, 2)
    p = m.predict_proba(x)
    assert p.shape == (5, 2) and np.allclose(p.sum(1), 1.0, atol=1e-5)
    assert m.quantum_features(x).shape == (5, N_QUBITS)


def test_model_can_overfit_two_clusters():
    torch.manual_seed(0)
    X = np.zeros((20, 512), dtype=np.float32)
    X[10:] = 1.0
    y = np.array([0] * 10 + [1] * 10)
    m = HybridModel(in_dim=512)
    fit(m, to_tensor(X), to_tensor(y, torch.long), epochs=150, verbose=False)
    assert (m.predict_proba(to_tensor(X)).argmax(1) == y).mean() >= 0.95


def test_classical_mode_has_no_qlayer():
    m = HybridModel(in_dim=512, use_circuit=False)
    assert m.qlayer is None
    assert m(to_tensor(np.random.randn(4, 512))).shape == (4, 2)


def test_layer_trace_matches_model_output():
    """The per-layer view shown in the app must end exactly at the model's quantum features."""
    torch.manual_seed(0)
    for ent in ("ring", "linear"):
        m = HybridModel(in_dim=512, entanglement=ent)
        m.eval()
        x = to_tensor(np.random.randn(3, 512))
        with torch.no_grad():
            trace = layer_trace(m.encode(x), m.qlayer.weights.detach(), m.n_qubits,
                                m.n_layers, m.entanglement)
        assert trace.shape == (m.n_layers, 3, m.n_qubits)
        assert np.allclose(trace[-1], m.quantum_features(x), atol=1e-5)


# ---------------------------------------------------------- experiments / mitigation
def test_qfeatures_matches_noisy_model():
    """Two code paths (helper vs. a model built with noise) must agree exactly."""
    from experiments import qfeatures, noise_spec
    torch.manual_seed(0)
    ideal = HybridModel(in_dim=512)
    ideal.eval()
    x = np.random.randn(5, 512).astype(np.float32)
    noise = noise_spec("phase_flip", 0.1)
    noisy = HybridModel(in_dim=512, noise=noise)
    noisy.load_state_dict(ideal.state_dict())
    assert np.allclose(qfeatures(ideal, x, noise), noisy.quantum_features(to_tensor(x)), atol=1e-5)
    assert np.allclose(qfeatures(ideal, x), ideal.quantum_features(to_tensor(x)), atol=1e-5)


def test_noise_spec():
    from experiments import noise_spec
    assert noise_spec("bit_flip", 0.0) is None
    assert noise_spec("bit_flip", 0.1) == {"type": "bit_flip", "p": 0.1}


def test_zne_extrapolation_is_exact_on_polynomials():
    from phase4 import extrapolate_to_zero
    lam = np.array([1.0, 3.0, 5.0])
    lin = 0.8 - 0.05 * lam
    quad = 0.8 - 0.05 * lam + 0.004 * lam ** 2
    assert np.allclose(extrapolate_to_zero(lam, list(lin), 1), 0.8)          # line through 2 points
    assert np.allclose(extrapolate_to_zero(lam, list(quad), 2), 0.8)         # Richardson, 3 points
    assert extrapolate_to_zero(lam, [2.0, 3.0, 4.0], 1) <= 1.0               # clipped to [-1, 1]


def test_threshold_recalibration_undoes_a_shift():
    from phase4 import fit_threshold
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    centred = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.3, 200)
    assert abs(fit_threshold(centred, y)) < 0.5                # already centred: stays near 0
    assert abs(fit_threshold(centred + 2.0, y) - 2.0) < 0.5   # shifted scores: threshold follows


def test_paired_statistics():
    from experiments import paired, mean_sd
    d, ci, wins, n = paired({0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.6, 1: 0.7, 2: 0.8})
    assert abs(d - 0.1) < 1e-9 and wins == 3 and n == 3 and ci >= 0
    m, s = mean_sd([1.0, 3.0])
    assert m == 2.0 and abs(s - np.sqrt(2)) < 1e-9


# ------------------------------------------------------------------------- dataset
def test_mask_and_tiny_images_are_rejected():
    from build_dataset import load_photo
    import tempfile
    from PIL import Image
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        photo = os.path.join(d, "photo.jpg")
        Image.fromarray(rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)).save(photo)
        mask = os.path.join(d, "mask.png")
        m = np.zeros((120, 120, 3), dtype=np.uint8)
        m[:40] = 1
        m[40:80] = 2
        Image.fromarray(m).save(mask)
        tiny = os.path.join(d, "tiny.jpg")
        Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)).save(tiny)
        broken = os.path.join(d, "broken.png")
        open(broken, "wb").write(b"not an image")
        assert load_photo(photo) is not None
        assert load_photo(mask) is None
        assert load_photo(tiny) is None
        assert load_photo(broken) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} tests passed")
