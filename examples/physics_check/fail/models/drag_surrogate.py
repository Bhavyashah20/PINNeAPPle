"""A stand-in for a drag surrogate Cd(Re, aoa) fitted on wind-tunnel points (Re 1e5 .. 1e6, aoa 0 .. 10 deg)."""
import numpy as np


def predict(X):
    re, aoa = X[:, 0], X[:, 1]
    return (0.006 + 0.9 / np.sqrt(re) * 10 + 1.5e-4 * aoa ** 2).reshape(-1, 1)
