"""OpenFOAM lduMatrix systems in Python: read the systems written by ``pinneWarmStart`` and convert them.

OpenFOAM stores a matrix over a mesh as three coefficient arrays and an addressing (``lduMatrix``):
``diag`` (one per cell), ``upper`` and ``lower`` (one per internal face), and for each internal face the owner
cell ``lowerAddr`` and the neighbour cell ``upperAddr``. The product ``A x`` is

    (A x)[c]  = diag[c] x[c]
    (A x)[l] += upper[f] x[u]      for every face f with l = lowerAddr[f], u = upperAddr[f]
    (A x)[u] += lower[f] x[l]

so ``A[l, u] = upper[f]`` and ``A[u, l] = lower[f]`` (equal for the symmetric pressure matrix). Boundary
conditions that are not coupled are already folded into ``diag`` and the source when the solver is called;
coupled interfaces (processor, cyclic) are not part of the arrays, which is why a dump records how many there
are (:attr:`LduSystem.n_coupled`) and :meth:`LduSystem.to_csr` refuses a system with coupled interfaces.

The stopping test of OpenFOAM's iterative solvers is a normalised L1 residual (``lduMatrix::solver::normFactor``),
reproduced by :func:`normalised_residual` so that Python solvers stop at the same point an OpenFOAM solver would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SMALL = 1e-20  # solverPerformance::small_ in double precision


@dataclass
class LduSystem:
    """One linear system ``A x = b`` as OpenFOAM handed it to the linear solver."""

    lower_addr: np.ndarray  # (nFaces,) int, owner cell of each internal face
    upper_addr: np.ndarray  # (nFaces,) int, neighbour cell
    diag: np.ndarray  # (nCells,)
    upper: np.ndarray  # (nFaces,)
    source: np.ndarray  # (nCells,) right-hand side b
    psi0: np.ndarray | None = None  # the incoming field (the guess a plain run starts from)
    psi: np.ndarray | None = None  # OpenFOAM's converged solution
    lower: np.ndarray | None = None  # (nFaces,) for an asymmetric matrix; None = symmetric
    meta: dict = field(default_factory=dict)
    _csr: object = field(default=None, repr=False, compare=False)

    @property
    def n_cells(self) -> int:
        return int(self.diag.size)

    @property
    def n_faces(self) -> int:
        return int(self.upper.size)

    @property
    def n_coupled(self) -> int:
        return int(self.meta.get("nCoupledInterfaces", 0))

    def amul(self, x: np.ndarray) -> np.ndarray:
        """``A x`` (also for (nCells, k) blocks): a cached CSR product, or the lduMatrix face loops."""
        if not self.n_coupled:
            if self._csr is None:
                self._csr = self.to_csr()
            return self._csr @ x
        return self.amul_ldu(x)

    def amul_ldu(self, x: np.ndarray) -> np.ndarray:
        """``A x`` with the lduMatrix face loops (vectorised)."""
        lo = self.upper if self.lower is None else self.lower
        y = self.diag.reshape((-1,) + (1,) * (x.ndim - 1)) * x
        up = self.upper.reshape((-1,) + (1,) * (x.ndim - 1))
        lw = lo.reshape((-1,) + (1,) * (x.ndim - 1))
        np.add.at(y, self.lower_addr, up * x[self.upper_addr])
        np.add.at(y, self.upper_addr, lw * x[self.lower_addr])
        return y

    def to_csr(self):
        """The matrix as ``scipy.sparse.csr_matrix`` (only for systems without coupled interfaces)."""
        import scipy.sparse as sp

        if self.n_coupled:
            raise ValueError(
                f"system has {self.n_coupled} coupled interfaces; the LDU arrays alone are not A"
            )
        lo = self.upper if self.lower is None else self.lower
        n = self.n_cells
        rows = np.concatenate([np.arange(n), self.lower_addr, self.upper_addr])
        cols = np.concatenate([np.arange(n), self.upper_addr, self.lower_addr])
        vals = np.concatenate([self.diag, self.upper, lo])
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def to_torch(self, device=None, dtype=None):
        """The matrix as a ``torch.sparse_csr_tensor`` (float64 by default)."""
        import warnings

        import torch

        A = self.to_csr()
        dtype = dtype or torch.float64
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "sparse CSR support is in beta"
            return torch.sparse_csr_tensor(
                torch.as_tensor(A.indptr, dtype=torch.int64),
                torch.as_tensor(A.indices, dtype=torch.int64),
                torch.as_tensor(A.data, dtype=dtype),
                size=A.shape,
                device=device,
            )


def from_csr(A, b: np.ndarray) -> LduSystem:
    """An :class:`LduSystem` from a symmetric sparse matrix (upper triangle = faces), for tests and synthetic cases."""
    import scipy.sparse as sp

    A = sp.coo_matrix(A)
    m = A.row < A.col
    order = np.lexsort((A.col[m], A.row[m]))
    return LduSystem(
        lower_addr=A.row[m][order].astype(np.int64),
        upper_addr=A.col[m][order].astype(np.int64),
        diag=np.asarray(sp.csr_matrix(A).diagonal(), dtype=float),
        upper=A.data[m][order].astype(float),
        source=np.asarray(b, dtype=float),
    )


def norm_factor(system: LduSystem, x: np.ndarray, Ax: np.ndarray | None = None) -> float:
    """OpenFOAM's residual normalisation: ``sum(|A x - A1 xbar| + |b - A1 xbar|) + small``."""
    Ax = system.amul(x) if Ax is None else Ax
    sum_a = system.amul(np.ones_like(x))
    tmp = sum_a * float(np.mean(x))
    return float(np.sum(np.abs(Ax - tmp) + np.abs(system.source - tmp))) + SMALL


def normalised_residual(system: LduSystem, x: np.ndarray, nf: float | None = None) -> float:
    """``sum|b - A x| / normFactor`` -- the residual OpenFOAM prints and tests against ``tolerance``."""
    Ax = system.amul(x)
    nf = norm_factor(system, x, Ax) if nf is None else nf
    return float(np.sum(np.abs(system.source - Ax))) / nf


def _read_meta(path: Path) -> dict:
    meta: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(" ")
        v = v.strip()
        try:
            meta[k] = int(v)
        except ValueError:
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v
    return meta


def read_system(directory) -> LduSystem:
    """Read one system written by ``pinneWarmStart`` (``dumpDir``): ``meta.txt`` and ``system.bin``."""
    d = Path(directory)
    meta = _read_meta(d / "meta.txt")
    nc, nf = int(meta["nCells"]), int(meta["nFaces"])
    raw = (d / "system.bin").read_bytes()
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=pos)
        pos += a.nbytes
        return a.astype(np.int64 if dtype == "<i4" else float)

    sysm = LduSystem(
        lower_addr=take("<i4", nf),
        upper_addr=take("<i4", nf),
        diag=take("<f8", nc),
        upper=take("<f8", nf),
        source=take("<f8", nc),
        psi0=take("<f8", nc),
        psi=take("<f8", nc),
        meta=meta,
    )
    if pos != len(raw):
        raise ValueError(f"{d}: {len(raw) - pos} unexpected trailing bytes")
    return sysm


def read_dump(dump_dir, field_name: str = "p") -> list[LduSystem]:
    """All systems of ``field_name`` in a dump directory, in solve order."""
    dirs = [
        p
        for p in Path(dump_dir).iterdir()
        if p.is_dir()
        and p.name.startswith(field_name + "_")
        and p.name[len(field_name) + 1 :].isdigit()
    ]
    dirs.sort(key=lambda p: int(p.name[len(field_name) + 1 :]))
    return [read_system(p) for p in dirs]
