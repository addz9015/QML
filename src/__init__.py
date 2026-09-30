# Import PennyLane before anything imports scipy / scikit-learn.
# On some Windows setups where `jax` is installed, jaxlib's native DLLs fail to load if
# scipy has already been imported; PennyLane touches jax on import. Loading it first makes
# the import order deterministic. (Harmless when jax is not installed.)
import pennylane as _pennylane  # noqa: F401
