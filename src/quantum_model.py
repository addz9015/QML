"""4-qubit variational quantum circuit (PennyLane) with data re-uploading.

Each layer:
  1. re-uploads the input as RY rotations           (data re-uploading)
  2. applies trainable RY, RZ on every qubit
  3. entangles with CNOTs                            (ring | linear | none)

Measurement: <Z_i> for each qubit -> n_qubits features.

Noise model (Phase 3+): after every *executed* layer, one channel (bit-flip,
phase-flip or depolarizing, strength p) is applied to every qubit. Simulated
exactly with the density-matrix device `default.mixed`.

Gate folding (Phase 4, zero-noise extrapolation): with `fold=k` each layer L is
executed as  L (L^dagger L)^k  with a noise channel after every application.
The ideal circuit is unchanged (L^dagger L = identity) but the noise is
amplified by a factor 2k+1 - exactly what folding does on real hardware.
"""

import numpy as np
import pennylane as qml

N_QUBITS = 4
N_LAYERS = 3
ENTANGLEMENTS = ("ring", "linear", "none")

# noise name -> PennyLane channel
_NOISE_CHANNELS = {
    "bit_flip": qml.BitFlip,
    "phase_flip": qml.PhaseFlip,
    "depolarizing": qml.DepolarizingChannel,
}
NOISE_TYPES = tuple(_NOISE_CHANNELS)


def _entangle(n_qubits, entanglement):
    if entanglement == "ring":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif entanglement == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif entanglement == "none":
        pass
    else:
        raise ValueError(f"unknown entanglement {entanglement!r}; use one of {ENTANGLEMENTS}")


def _apply_layer(inputs, weights, layer, n_qubits, entanglement):
    for i in range(n_qubits):                               # data re-uploading
        qml.RY(inputs[..., i], wires=i)
    for i in range(n_qubits):                               # trainable rotations
        qml.RY(weights[layer, i, 0], wires=i)
        qml.RZ(weights[layer, i, 1], wires=i)
    _entangle(n_qubits, entanglement)


def make_qnode(n_qubits=N_QUBITS, n_layers=N_LAYERS, noise=None,
               entanglement="ring", fold=0):
    """Build the QNode.

    noise = {"type": "depolarizing", "p": 0.05} or None (ideal, `default.qubit`).
    fold  = number of (L^dagger L) folds per layer (noise amplification 2*fold+1).
    """
    if entanglement not in ENTANGLEMENTS:
        raise ValueError(f"unknown entanglement {entanglement!r}")

    if noise:
        dev = qml.device("default.mixed", wires=n_qubits)
        channel = _NOISE_CHANNELS[noise["type"]]
        noise_p = noise["p"]
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
        channel = None

    def _noise():
        if channel is not None:
            for i in range(n_qubits):
                channel(noise_p, wires=i)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            _apply_layer(inputs, weights, layer, n_qubits, entanglement)
            _noise()
            for _ in range(fold):
                qml.adjoint(_apply_layer)(inputs, weights, layer, n_qubits, entanglement)
                _noise()
                _apply_layer(inputs, weights, layer, n_qubits, entanglement)
                _noise()
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def make_hardware_qnode(dev, n_qubits=N_QUBITS, n_layers=N_LAYERS, entanglement="ring", fold=0):
    """Same circuit as make_qnode, but on a caller-supplied device (Phase 5: a real IBM
    backend, or a local fake/noise-model backend for testing). No diff_method: hardware and
    fake backends don't support exact gradients, and Phase 5 is inference only."""

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for layer in range(n_layers):
            _apply_layer(inputs, weights, layer, n_qubits, entanglement)
            for _ in range(fold):
                qml.adjoint(_apply_layer)(inputs, weights, layer, n_qubits, entanglement)
                _apply_layer(inputs, weights, layer, n_qubits, entanglement)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def _qiskit_layer_circuit(angles, weights, layer, n_qubits, entanglement):
    """One circuit layer as a plain Qiskit QuantumCircuit (for Phase 5 hardware runs)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(angles[i]), i)
    for i in range(n_qubits):
        qc.ry(float(weights[layer, i, 0]), i)
        qc.rz(float(weights[layer, i, 1]), i)
    if entanglement == "ring":
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    elif entanglement == "linear":
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    elif entanglement != "none":
        raise ValueError(f"unknown entanglement {entanglement!r}; use one of {ENTANGLEMENTS}")
    return qc


def build_qiskit_circuit(angles, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                         entanglement="ring", fold=0):
    """The same circuit as make_qnode's ideal case, as a plain Qiskit QuantumCircuit,
    for one fixed (numeric) input. Used to submit Phase 5 hardware jobs directly."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        block = _qiskit_layer_circuit(angles, weights, layer, n_qubits, entanglement)
        qc.compose(block, inplace=True)
        for _ in range(fold):
            qc.compose(block.inverse(), inplace=True)
            qc.compose(block, inplace=True)
    return qc


def run_on_backend(backend, angles_batch, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                   entanglement="ring", fold=0, shots=1024):
    """Run the circuit for every row of angles_batch (batch, n_qubits) on a real (or local
    fake) Qiskit backend, as separate pubs in ONE Runtime job - not one job per image.

    `fold` is a single gate-folding level applied to every image, or an array-like with one
    level per row of angles_batch (mixed-fold jobs - e.g. Phase 5's ZNE step, which submits
    several fold levels for the same images together instead of one job per level).

    Returns ((batch, n_qubits) array of <Z_i>, the Job object)."""
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    z_obs = [SparsePauliOp.from_list([("I" * (n_qubits - 1 - i) + "Z" + "I" * i, 1)])
             for i in range(n_qubits)]
    folds = np.broadcast_to(fold, (len(angles_batch),))

    pubs = []
    for angles, f in zip(angles_batch, folds):
        isa_circuit = pm.run(build_qiskit_circuit(angles, weights, n_qubits, n_layers,
                                                   entanglement, int(f)))
        isa_obs = [o.apply_layout(isa_circuit.layout) for o in z_obs]
        pubs.append((isa_circuit, isa_obs))

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    job = estimator.run(pubs)
    result = job.result()
    return np.array([r.data.evs for r in result], dtype=float), job


def make_torch_layer(n_qubits=N_QUBITS, n_layers=N_LAYERS, noise=None, entanglement="ring"):
    qnode = make_qnode(n_qubits, n_layers, noise, entanglement)
    return qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 2)})


def run_qnode(qnode, inputs, weights):
    """Run a QNode on a batch and return a (batch, n_qubits) float array."""
    out = qnode(inputs, weights)
    return np.stack([np.asarray(o.detach().cpu() if hasattr(o, "detach") else o, dtype=float)
                     for o in out], axis=-1)


def layer_trace(inputs, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                entanglement="ring", noise=None):
    """<Z_i> measured after each layer -> array (n_layers, batch, n_qubits).

    Measurements can only be taken at the end of a circuit, so layer k is obtained
    by running the first k layers (sharing the first k rows of `weights`).
    The last entry equals what the full circuit outputs.
    """
    return np.stack([
        run_qnode(make_qnode(n_qubits, k, noise, entanglement), inputs, weights[:k])
        for k in range(1, n_layers + 1)
    ])


def draw_circuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, entanglement="ring"):
    """ASCII drawing of the circuit (symbolic inputs / weights)."""
    import torch
    qnode = make_qnode(n_qubits, n_layers, None, entanglement)
    return qml.draw(qnode, level="top")(torch.zeros(1, n_qubits),
                                        torch.zeros(n_layers, n_qubits, 2))
