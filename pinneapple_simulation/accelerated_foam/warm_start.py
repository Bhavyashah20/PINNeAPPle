"""Initial guesses for a sequence of pressure systems.

A transient pressure-velocity solver solves a new pressure system every corrector, with a matrix and a
right-hand side that change slowly. OpenFOAM starts each solve from the current pressure field. The guesses
here are never worse than that in the energy norm of the current matrix, because that field is in the space
they project on:

- :class:`ProjectionWarmStart` -- Galerkin projection on the last ``k`` solutions (Fischer 1998). This is the
  algorithm of the ``pinneWarmStart`` OpenFOAM wrapper (``guess projection``); the Python version is used
  for the offline benchmarks and to check the C++ one.
- :class:`PODWarmStart` -- Galerkin projection on a POD basis of solutions from earlier runs (a design of
  experiments, a previous simulation), plus the current field.

Both use only ``A x`` products, so they apply to any matrix the solver hands over.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .ldu import LduSystem


def galerkin(system: LduSystem, basis: np.ndarray, drop: float = 1e-12) -> np.ndarray | None:
    """``x = V c`` with ``(V^T A V) c = V^T b``; dependent columns of ``V`` (n, k) are dropped. None if empty."""
    if basis.size == 0:
        return None
    AV = system.amul(basis)
    G = basis.T @ AV
    r = basis.T @ system.source
    # symmetric elimination with diagonal pivoting, like the C++ wrapper
    k = G.shape[0]
    M, y = G.copy(), r.copy()
    dmax = float(np.max(np.abs(np.diag(G))))
    if dmax <= 0:
        return None
    active = np.ones(k, bool)
    order: list[int] = []
    for _ in range(k):
        cand = np.where(active)[0]
        p = int(cand[np.argmax(np.abs(np.diag(M)[cand]))])
        if abs(M[p, p]) <= drop * dmax:
            break
        active[p] = False
        order.append(p)
        for i in np.where(active)[0]:
            f = M[i, p] / M[p, p]
            M[i] -= f * M[p]
            y[i] -= f * y[p]
    if not order:
        return None
    coef = np.zeros(k)
    for o in range(len(order) - 1, -1, -1):
        p = order[o]
        s = y[p] - sum(M[p, order[q]] * coef[order[q]] for q in range(o + 1, len(order)))
        coef[p] = s / M[p, p]
    return basis @ coef


class ProjectionWarmStart:
    """Galerkin projection of each new system on the span of the last ``n_snapshots`` solutions and of the
    incoming field. Call :meth:`guess` before a solve and :meth:`update` with the converged solution after it."""

    def __init__(self, n_snapshots: int = 8):
        self.n_snapshots = int(n_snapshots)
        self.snapshots: deque[np.ndarray] = deque(maxlen=self.n_snapshots)

    def guess(self, system: LduSystem, x_prev: np.ndarray) -> np.ndarray:
        cols = list(self.snapshots)
        if not cols or not np.array_equal(cols[-1], x_prev):
            cols.append(np.asarray(x_prev, float))
        x = galerkin(system, np.stack(cols, axis=1))
        return np.asarray(x_prev, float) if x is None else x

    def update(self, x: np.ndarray) -> None:
        self.snapshots.append(np.array(x, dtype=float))


class PODWarmStart:
    """Galerkin projection on a POD basis of stored solutions (other runs or earlier steps) and the current field.

    ``fit`` keeps the left singular vectors that hold ``energy`` of the snapshot energy (at most ``max_rank``).
    With ``n_recent > 0`` the last ``n_recent`` solutions of the current run join the basis (POD of earlier
    runs + projection on the current one).
    """

    def __init__(self, energy: float = 0.99999, max_rank: int = 32, n_recent: int = 0):
        self.energy = float(energy)
        self.max_rank = int(max_rank)
        self.basis: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.recent: deque[np.ndarray] = deque(maxlen=max(int(n_recent), 1))
        self.n_recent = int(n_recent)

    def fit(self, snapshots: np.ndarray) -> PODWarmStart:
        """``snapshots``: (n_cells, n_snapshots)."""
        U, S, _ = np.linalg.svd(np.asarray(snapshots, float), full_matrices=False)
        e = np.cumsum(S**2) / max(float(np.sum(S**2)), 1e-300)
        r = int(min(self.max_rank, np.searchsorted(e, self.energy) + 1, U.shape[1]))
        self.basis, self.singular_values = U[:, :r], S
        return self

    @property
    def rank(self) -> int:
        return 0 if self.basis is None else int(self.basis.shape[1])

    def guess(self, system: LduSystem, x_prev: np.ndarray) -> np.ndarray:
        cols = [] if self.basis is None else [self.basis]
        if self.n_recent:
            cols += [v[:, None] for v in self.recent]
        cols.append(np.asarray(x_prev, float)[:, None])
        x = galerkin(system, np.concatenate(cols, axis=1))
        return np.asarray(x_prev, float) if x is None else x

    def update(self, x: np.ndarray) -> None:
        """The basis is fitted offline; only the recent solutions (``n_recent``) are kept per solve."""
        if self.n_recent:
            self.recent.append(np.array(x, dtype=float))
