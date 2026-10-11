"""PT-symmetric dimers and chains: Hamiltonian, spectrum, exceptional points (EPs) and how they coalesce.

The dimer

    H(gamma, lam) = [[eps + i gamma,  b_ab(lam)],
                     [b_ba(lam),      eps - i gamma]]

is PT symmetric (P = sigma_x, T = complex conjugation, P H* P = H) when ``b_ba = conj(b_ab)`` (reciprocal coupling).
Its eigenvalues are ``eps +/- sqrt(b_ab b_ba - gamma^2)``, so the eigenvalues *and* the eigenvectors coalesce at
``gamma_EP^2 = b_ab b_ba`` (= |b|^2 in the PT-symmetric case): an exceptional point. Below it the spectrum is real
(unbroken PT phase), above it complex conjugate (broken phase).

:class:`PTDimer` evaluates all of this for couplings given as numbers or functions of a control parameter ``lam``;
:meth:`PTDimer.ep_gamma_numeric` finds the EP from the matrix alone (eigenvalue splitting and eigenvector overlap),
an independent check of the closed form. :class:`PTChain` does the same for N sites (gain on the first, loss on the
last), where the EP can be of higher order (the trimer's is an EP3 at gamma = sqrt(2) |b|).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["PTDimer", "PTChain", "pt_violation", "eigenvector_overlap", "petermann_factor"]

Coupling = complex | float | Callable[[float], complex]


def _as_fn(c: Coupling | None) -> Callable[[float], complex] | None:
    if c is None:
        return None
    if callable(c):
        return lambda lam: complex(c(lam))
    return lambda lam, v=complex(c): v


def pt_violation(H: np.ndarray) -> float:
    """||P H* P - H|| / ||H|| with P the exchange (anti-identity) matrix: 0 for a PT-symmetric Hamiltonian."""
    H = np.asarray(H, dtype=complex)
    P = np.fliplr(np.eye(H.shape[0]))
    return float(np.linalg.norm(P @ H.conj() @ P - H) / max(np.linalg.norm(H), 1e-300))


def eigenvector_overlap(H: np.ndarray) -> float:
    """Largest |<v_i|v_j>| / (|v_i||v_j|) over pairs of right eigenvectors: 1 when two eigenvectors coalesce (EP),
    small for well-separated modes of a nearly Hermitian matrix."""
    _, V = np.linalg.eig(np.asarray(H, dtype=complex))
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    G = np.abs(V.conj().T @ V)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def petermann_factor(H: np.ndarray) -> np.ndarray:
    """K_i = <L_i|L_i><R_i|R_i> / |<L_i|R_i>|^2 for each mode (1 for a Hermitian matrix, diverges at an EP)."""
    w, R = np.linalg.eig(np.asarray(H, dtype=complex))
    L = np.linalg.inv(R).conj().T                                     # left eigenvectors as columns
    num = np.sum(np.abs(L) ** 2, 0) * np.sum(np.abs(R) ** 2, 0)
    den = np.abs(np.sum(L.conj() * R, 0)) ** 2
    return num / np.maximum(den, 1e-300)


@dataclass
class PTDimer:
    """Two-site gain/loss system. ``coupling``: b_ab as a number or a function of the control parameter ``lam``;
    ``coupling_ba``: b_ba (default ``conj(b_ab)``: reciprocal, PT symmetric; give it to model non-reciprocal
    hopping); ``onsite``: common on-site energy eps."""
    coupling: Coupling = 1.0
    coupling_ba: Coupling | None = None
    onsite: float = 0.0

    def __post_init__(self):
        self._ab = _as_fn(self.coupling)
        self._ba = _as_fn(self.coupling_ba)

    def b(self, lam: float = 0.0) -> tuple[complex, complex]:
        ab = self._ab(lam)
        ba = self._ba(lam) if self._ba is not None else ab.conjugate()
        return ab, ba

    def hamiltonian(self, gamma: float, lam: float = 0.0) -> np.ndarray:
        ab, ba = self.b(lam)
        return np.array([[self.onsite + 1j * gamma, ab], [ba, self.onsite - 1j * gamma]], dtype=complex)

    def is_pt_symmetric(self, gamma: float = 0.5, lam: float = 0.0, tol: float = 1e-12) -> bool:
        return pt_violation(self.hamiltonian(gamma, lam)) <= tol

    def eigenvalues(self, gamma: float, lam: float = 0.0) -> np.ndarray:
        """Closed form eps -/+ sqrt(b_ab b_ba - gamma^2), sorted by real then imaginary part."""
        ab, ba = self.b(lam)
        s = np.sqrt(complex(ab * ba - gamma ** 2))
        return np.array(sorted([self.onsite - s, self.onsite + s], key=lambda z: (z.real, z.imag)))

    def ep_gamma_squared(self, lam: float = 0.0) -> complex:
        """gamma_EP^2 = b_ab b_ba (|b|^2 when PT symmetric). An EP at real gamma needs it real and positive."""
        ab, ba = self.b(lam)
        return complex(ab * ba)

    def ep_gamma(self, lam: float = 0.0) -> float:
        g2 = self.ep_gamma_squared(lam)
        if abs(g2.imag) > 1e-12 * max(1.0, abs(g2)) or g2.real < 0:
            return math.nan                                           # no EP on the real gamma axis
        return math.sqrt(g2.real)

    def phase(self, gamma: float, lam: float = 0.0, tol: float = 1e-9) -> str:
        """``"unbroken"`` (real spectrum), ``"broken"`` (complex conjugate pair) or ``"EP"``."""
        g_ep = self.ep_gamma(lam)
        if not np.isfinite(g_ep):
            return "broken"
        if abs(gamma - g_ep) <= tol * max(1.0, g_ep):
            return "EP"
        return "unbroken" if gamma < g_ep else "broken"

    def ep_gamma_numeric(self, lam: float = 0.0, gamma_max: float | None = None, tol: float = 1e-10) -> float:
        """The EP from the matrix alone: the gamma where the eigenvalue splitting first becomes purely imaginary,
        found by bisection on the numerical spectrum (no closed form used)."""
        def broken(g):
            w = np.linalg.eigvals(self.hamiltonian(g, lam))
            return abs(w[0].imag - w[1].imag) > 1e-7 * max(1.0, abs(w[0] - w[1]))
        hi = gamma_max or 2.0 * max(abs(self.b(lam)[0]), abs(self.b(lam)[1]), 1e-6) + 1.0
        lo = 0.0
        if broken(lo):
            return math.nan
        if not broken(hi):
            return math.nan
        while hi - lo > tol * max(1.0, hi):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if broken(mid) else (mid, hi)
        return 0.5 * (lo + hi)

    def eigenvector_overlap(self, gamma: float, lam: float = 0.0) -> float:
        return eigenvector_overlap(self.hamiltonian(gamma, lam))

    def splitting(self, gamma: float, lam: float = 0.0) -> float:
        """|E+ - E-| = 2 |sqrt(b_ab b_ba - gamma^2)|: vanishes at the EP as the square root of the distance to it,
        the origin of EP-enhanced sensitivity."""
        w = self.eigenvalues(gamma, lam)
        return float(abs(w[1] - w[0]))


@dataclass
class PTChain:
    """N-site chain with gain +i gamma on the first site, loss -i gamma on the last and nearest-neighbour coupling
    ``b`` (number or list of N-1 numbers; reciprocal unless ``reciprocal=False`` and ``b_back`` is given)."""
    n: int
    b: complex | Sequence[complex] = 1.0
    b_back: complex | Sequence[complex] | None = None

    def _bonds(self):
        bb = np.broadcast_to(np.asarray(self.b, dtype=complex), (self.n - 1,))
        back = np.conj(bb) if self.b_back is None else np.broadcast_to(np.asarray(self.b_back, dtype=complex), (self.n - 1,))
        return bb, back

    def hamiltonian(self, gamma: float) -> np.ndarray:
        H = np.zeros((self.n, self.n), dtype=complex)
        bb, back = self._bonds()
        for i in range(self.n - 1):
            H[i, i + 1], H[i + 1, i] = bb[i], back[i]
        H[0, 0], H[-1, -1] = 1j * gamma, -1j * gamma
        return H

    def ep_gamma_numeric(self, gamma_max: float | None = None, tol: float = 1e-9) -> float:
        """Smallest gamma at which the spectrum stops being real (PT-breaking threshold, an EP)."""
        def broken(g):
            return np.max(np.abs(np.linalg.eigvals(self.hamiltonian(g)).imag)) > 1e-6
        hi = gamma_max or 4.0 * float(np.max(np.abs(self._bonds()[0]))) + 1.0
        lo = 0.0
        if broken(lo) or not broken(hi):
            return math.nan
        while hi - lo > tol * max(1.0, hi):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if broken(mid) else (mid, hi)
        return 0.5 * (lo + hi)

    def ep_order(self, gamma: float, tol: float = 1e-3) -> int:
        """How many eigenvalues coalesce near ``gamma`` (2 for an ordinary EP, 3 for the trimer's EP3...)."""
        w = np.linalg.eigvals(self.hamiltonian(gamma))
        scale = max(1.0, float(np.max(np.abs(w))))
        best = 1
        for z in w:
            best = max(best, int(np.sum(np.abs(w - z) < tol * scale)))
        return best
