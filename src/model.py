"""End-to-end differentiable hybrid model.

    ResNet-512 features
      -> Linear(512 -> n_qubits)        trainable supervised projection (replaces PCA)
      -> BatchNorm + learnable scale    keeps encoding angles well-conditioned
      -> quantum circuit (TorchLayer)   data re-uploading + ring entanglement
      -> Linear(n_qubits -> 2)          trainable readout
      -> logits

Projection, circuit and readout are trained jointly with Adam / cross-entropy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from quantum_model import make_torch_layer, N_QUBITS, N_LAYERS


class HybridModel(nn.Module):

    def __init__(self, in_dim=512, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                 freeze_circuit=False, use_circuit=True, noise=None,
                 entanglement="ring"):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_circuit = use_circuit
        self.entanglement = entanglement

        self.proj = nn.Linear(in_dim, n_qubits)
        self.bn = nn.BatchNorm1d(n_qubits)
        self.scale = nn.Parameter(torch.ones(1))
        self.qlayer = (make_torch_layer(n_qubits, n_layers, noise, entanglement)
                       if use_circuit else None)
        self.head = nn.Linear(n_qubits, 2)

        if freeze_circuit and self.qlayer is not None:
            for p in self.qlayer.parameters():
                p.requires_grad_(False)

    def encode(self, x):
        return self.scale * self.bn(self.proj(x))

    def _q(self, z):
        return self.qlayer(z) if self.use_circuit else torch.tanh(z)

    def forward(self, x):
        return self.head(self._q(self.encode(x)))

    @torch.no_grad()
    def quantum_features(self, x):
        self.eval()
        return self._q(self.encode(x)).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self.forward(x), dim=1).cpu().numpy()


def _acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


def fit(model, X_train, y_train, X_val=None, y_val=None,
        epochs=150, lr=0.05, weight_decay=1e-4, seed=42, verbose=True):
    """Full-batch Adam training. Restores the best-validation weights at the end."""

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    have_val = X_val is not None
    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

        if have_val:
            va = _acc(model, X_val, y_val)
            if va >= best_acc:
                best_acc = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 25 == 0 or epoch == 1):
            tr = _acc(model, X_train, y_train)
            msg = f"  epoch {epoch:3d}  loss={float(loss):.4f}  train_acc={tr:.3f}"
            if have_val:
                msg += f"  val_acc={va:.3f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def to_tensor(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)
