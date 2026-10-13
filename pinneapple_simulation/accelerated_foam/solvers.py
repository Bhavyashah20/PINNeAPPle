"""Krylov solvers for extracted pressure systems, with OpenFOAM's stopping test.

Every solver stops when the normalised residual (:func:`ldu.normalised_residual`, normalisation computed at the
initial guess, as OpenFOAM does) is below ``tolerance`` or below ``rel_tol`` times its initial value, so an
iteration count here can be compared with an OpenFOAM log line at the same settings.

- :func:`pcg` -- preconditioned conjugate gradients in NumPy/SciPy, with
  :func:`jacobi`, :func:`dic` (OpenFOAM's diagonal incomplete Cholesky) or :func:`amg` (pyamg, optional) preconditioners.
- :func:`torch_pcg` -- the same Jacobi-preconditioned CG in PyTorch (GPU when one is available), in float64 or
  float32 with float64 residual replacement (mixed precision).

The pressure matrix of an incompressible solver is negative definite; CG is applied to it unchanged (the
preconditioners carry the sign).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ldu import LduSystem, norm_factor


@dataclass
class SolveResult:
    x: np.ndarray
    iterations: int
    initial_residual: float
    final_residual: float
    wall_time: float
    converged: bool
    history: list[float] = field(default_factory=list)
    setup_time: float = 0.0


# ------------------------------------------------------------------------------------------ preconditioners


def jacobi(system: LduSystem):
    rd = 1.0 / system.diag
    return lambda r: rd * r


def dic(system: LduSystem):
    """OpenFOAM's DIC: ``M = (D* + L) D*^-1 (D* + U)`` with ``D*`` from the face sweep, applied by two
    sparse triangular solves."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve_triangular

    n = system.n_cells
    lo, up = system.lower_addr, system.upper_addr
    coef = system.upper
    d = system.diag.astype(float).copy()
    # Sequential face sweep as in OpenFOAM: faces are ordered by owner, so d[l] is final when face (l, u) is met.
    for f in range(coef.size):
        d[up[f]] -= coef[f] * coef[f] / d[lo[f]]
    L = sp.csr_matrix((coef, (up, lo)), shape=(n, n)) + sp.diags(d)
    U = sp.csr_matrix((coef, (lo, up)), shape=(n, n)) + sp.diags(d)
    L, U = L.tocsr(), U.tocsr()

    def apply(r):
        y = spsolve_triangular(L, r, lower=True)
        return spsolve_triangular(U, d * y, lower=False)

    return apply


def amg(system: LduSystem, **kwargs):
    """Smoothed-aggregation AMG V-cycle from pyamg (MIT licence; optional) applied to ``-A`` (SPD)."""
    import pyamg

    A = system.to_csr()
    sign = -1.0 if np.mean(system.diag) < 0 else 1.0
    ml = pyamg.smoothed_aggregation_solver(sign * A, **kwargs)
    M = ml.aspreconditioner(cycle="V")
    return lambda r: sign * M.matvec(r)


PRECONDITIONERS = {"jacobi": jacobi, "dic": dic, "amg": amg}


# ------------------------------------------------------------------------------------------ CG


def pcg(
    system: LduSystem,
    x0: np.ndarray | None = None,
    preconditioner="dic",
    tolerance: float = 1e-6,
    rel_tol: float = 0.0,
    max_iter: int = 1000,
    min_iter: int = 0,
) -> SolveResult:
    """Preconditioned CG, OpenFOAM's PCG algorithm and stopping test, in NumPy."""
    t0 = time.perf_counter()
    M = (
        PRECONDITIONERS[preconditioner](system)
        if isinstance(preconditioner, str)
        else preconditioner
    )
    t_setup = time.perf_counter() - t0
    b = system.source
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    Ax = system.amul(x)
    nf = norm_factor(system, x, Ax)
    r = b - Ax
    res0 = float(np.sum(np.abs(r))) / nf
    res = res0
    hist = [res0]
    it = 0
    p = np.zeros_like(b)
    wr_old = np.inf
    while (
        it < min_iter or not (res < tolerance or (rel_tol > 0 and res < rel_tol * res0))
    ) and it < max_iter:
        w = M(r)
        wr = float(w @ r)
        beta = 0.0 if it == 0 else wr / wr_old
        p = w + beta * p
        Ap = system.amul(p)
        pAp = float(p @ Ap)
        if pAp == 0.0:
            break
        alpha = wr / pAp
        x += alpha * p
        r -= alpha * Ap
        wr_old = wr
        it += 1
        res = float(np.sum(np.abs(r))) / nf
        hist.append(res)
    conv = res < tolerance or (rel_tol > 0 and res < rel_tol * res0)
    return SolveResult(x, it, res0, res, time.perf_counter() - t0, conv, hist, t_setup)


def torch_pcg(
    system: LduSystem,
    x0: np.ndarray | None = None,
    tolerance: float = 1e-6,
    rel_tol: float = 0.0,
    max_iter: int = 1000,
    device=None,
    dtype: str = "float64",
    inner_rel: float = 1e-2,
) -> SolveResult:
    """Jacobi-preconditioned CG in PyTorch (GPU when available).

    ``dtype="float32"`` is mixed-precision iterative refinement: the solution and the residual are kept in
    float64, each correction ``A d = r`` is solved by float32 CG to ``inner_rel`` of its residual, and the
    stopping test is made on the float64 residual (Carson & Higham 2018). ``iterations`` counts the float32
    CG iterations. Meaningful on a GPU; on a CPU single precision is rarely faster.
    """
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    A64 = system.to_torch(device=dev, dtype=torch.float64)
    b = torch.as_tensor(system.source, device=dev, dtype=torch.float64)
    x = (
        torch.zeros_like(b)
        if x0 is None
        else torch.as_tensor(np.asarray(x0, float), device=dev).clone()
    )
    rd64 = 1.0 / torch.as_tensor(system.diag, device=dev, dtype=torch.float64)
    sum_a = A64 @ torch.ones_like(b)
    Ax = A64 @ x
    tmp = sum_a * x.mean()
    nf = float(torch.sum(torch.abs(Ax - tmp) + torch.abs(b - tmp))) + 1e-20
    r = b - Ax
    res0 = float(torch.sum(torch.abs(r))) / nf
    res = res0
    hist = [res0]

    def done(rv):
        return rv < tolerance or (rel_tol > 0 and rv < rel_tol * res0)

    def cg(A, rd, rhs, xs, stop, budget):
        """Plain Jacobi-PCG on A xs = rhs from xs; stop(res_vector) -> bool. Returns xs, iterations."""
        rr = rhs - A @ xs
        p = None
        wr_old = None
        k = 0
        while k < budget and not stop(rr):
            w = rd * rr
            wr = torch.dot(w, rr)
            p = w if p is None else w + (wr / wr_old) * p
            Ap = A @ p
            alpha = wr / torch.dot(p, Ap)
            xs = xs + alpha * p
            rr = rr - alpha * Ap
            wr_old = wr
            k += 1
        return xs, k

    it = 0
    if dtype == "float64":

        def stop64(rr):
            hist.append(float(torch.sum(torch.abs(rr))) / nf)
            return done(hist[-1])

        hist.pop()
        x, it = cg(A64, rd64, b, x, stop64, max_iter)
        res = float(torch.sum(torch.abs(b - A64 @ x))) / nf
    else:
        A32 = system.to_torch(device=dev, dtype=torch.float32)
        rd32 = rd64.to(torch.float32)
        while not done(res) and it < max_iter:
            r32 = r.to(torch.float32)
            target = inner_rel * float(torch.sum(torch.abs(r32)))
            d, k = cg(
                A32,
                rd32,
                r32,
                torch.zeros_like(r32),
                lambda rr, t=target: float(torch.sum(torch.abs(rr))) <= t,
                max_iter - it,
            )
            it += max(k, 1)
            x = x + d.to(torch.float64)
            r = b - A64 @ x
            res = float(torch.sum(torch.abs(r))) / nf
            hist.append(res)
            if k == 0:
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return SolveResult(x.cpu().numpy(), it, res0, res, time.perf_counter() - t0, done(res), hist)
