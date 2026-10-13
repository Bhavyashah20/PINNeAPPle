"""A stand-in for a PINN trained on Kovasznay flow (Re = 40): the exact solution plus a small smooth error, as a
trained network would have. ``predict`` takes (N, 2) inputs x, y and returns (N, 3) outputs u, v, p."""
import numpy as np

RE = 40.0
LAM = RE / 2 - np.sqrt(RE ** 2 / 4 + 4 * np.pi ** 2)


def predict(X):
    x, y = X[:, 0], X[:, 1]
    e = np.exp(LAM * x)
    u = (1 - e * np.cos(2 * np.pi * y)) * (1 + 0.0 * (x + 0.5))
    v = LAM / (2 * np.pi) * e * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * LAM * x))
    err = 1e-4 * np.sin(3 * x) * np.cos(2 * y)
    return np.column_stack([u + err, v - err, p + err])
