"""Dynamic Mode Decomposition (DMD) — batch-capable, stabilised, with spectral analysis."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .base import ROMBase, ROMOutput

CenterMode = bool | Literal["global", "sequence"]


class DynamicModeDecomposition(ROMBase):
    """
    Truncated (rank-r) DMD in reduced space.

    Accepts snapshots X with shape (T, D) or (B, T, D).

    Reduced operator
    ----------------
    X0 = U_r S_r V_r^T  (truncated SVD of left-snapshot matrix)
    A_tilde = U_r^T X1 V_r S_r^{-1}

    Ridge regularisation
    --------------------
    S_r^{-1}_ridge = diag(S / (S^2 + l2))   -- reduces to 1/S when l2=0

    Additional capabilities vs. the original
    -----------------------------------------
    - fit_pairs(X0, X1): accept explicit (N,D) snapshot pairs (used by HAVOK)
    - eig(): full spectral decomposition — eigenvalues, modes, frequencies
    - explained_variance_ratio_: energy fraction captured by each mode
    """

    def __init__(self, r: int = 64, center: CenterMode = "global", l2: float = 0.0):
        super().__init__()
        self.r = int(r)
        self.center: CenterMode = center
        self.l2 = float(l2)

        self.register_buffer("mean_",  torch.zeros(1))
        self.register_buffer("basis_", torch.zeros(1))
        self.register_buffer("A_",     torch.zeros(1))
        self._fitted = False
        self._singular_values: torch.Tensor | None = None  # (r,) for explained variance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_btd(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:
            return X[None]
        if X.ndim == 3:
            return X
        raise ValueError(f"Expected X with shape (T,D) or (B,T,D), got {tuple(X.shape)}")

    def _center_mean(self, Xseq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Xc, mu) based on self.center mode."""
        B, T, D = Xseq.shape
        if self.center is False:
            mu = torch.zeros((1, D), device=Xseq.device, dtype=Xseq.dtype)
            return Xseq, mu
        if self.center in ("global", True):
            mu = Xseq.reshape(B * T, D).mean(dim=0, keepdim=True)
            return Xseq - mu, mu
        if self.center == "sequence":
            mu_seq = Xseq.mean(dim=1, keepdim=True)
            Xc = Xseq - mu_seq
            mu = mu_seq.mean(dim=0).view(1, D)
            return Xc, mu
        raise ValueError('center must be False, "global", "sequence", or True.')

    @staticmethod
    def _build_operator(
        X0: torch.Tensor,  # (D, N)
        X1: torch.Tensor,  # (D, N)
        r: int,
        l2: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core DMD computation from paired snapshot matrices.
        Returns (Ur, A_tilde, S_r).
        """
        U, S, Vh = torch.linalg.svd(X0, full_matrices=False)
        r_eff = min(r, U.shape[1])
        Ur = U[:, :r_eff]
        Sr = S[:r_eff]
        Vr = Vh[:r_eff].t()

        eps = 1e-12
        inv = Sr / (Sr * Sr + l2) if l2 > 0.0 else 1.0 / Sr.clamp_min(eps)
        Sr_inv = torch.diag(inv)

        A_tilde = Ur.t() @ X1 @ Vr @ Sr_inv
        return Ur, A_tilde, Sr

    # ------------------------------------------------------------------
    # Fit API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> DynamicModeDecomposition:
        """Fit from sequential snapshots (T,D) or (B,T,D)."""
        Xseq = self._as_btd(X)
        B, T, D = Xseq.shape
        if T < 2:
            raise ValueError("Need at least T>=2 snapshots.")

        Xc, mu = self._center_mean(Xseq)
        self.mean_ = mu.detach().clone()          # keep the data's dtype and device

        X0 = Xc[:, :-1].reshape(-1, D).t()
        X1 = Xc[:, 1: ].reshape(-1, D).t()

        Ur, A_tilde, Sr = self._build_operator(X0, X1, self.r, self.l2)
        self.basis_ = Ur.detach().clone()          # keep the data's dtype and device
        self.A_ = A_tilde.detach().clone()          # keep the data's dtype and device
        self._singular_values = Sr.clone()
        self._fitted = True
        return self

    @torch.no_grad()
    def fit_pairs(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        *,
        center: bool = True,
    ) -> DynamicModeDecomposition:
        """
        Fit from explicit (N, D) snapshot pairs.

        Accepts both 2-D (N,D) and 3-D (B,T,D) inputs; in the latter case
        the batch/time dimensions are flattened into N.

        Parameters
        ----------
        X0, X1 : (N, D) or (B, T, D) paired snapshot matrices.
        center  : if True, subtract the global mean of X0 before fitting.
        """
        # Flatten to (N, D)
        if X0.ndim == 3:
            X0 = X0.reshape(-1, X0.shape[-1])
        if X1.ndim == 3:
            X1 = X1.reshape(-1, X1.shape[-1])

        if X0.shape != X1.shape:
            raise ValueError(f"X0 and X1 must have the same shape; got {X0.shape} vs {X1.shape}")

        N, D = X0.shape
        if center:
            mu = X0.mean(dim=0, keepdim=True)
            X0c = X0 - mu
            X1c = X1 - mu
        else:
            mu = torch.zeros((1, D), device=X0.device, dtype=X0.dtype)
            X0c, X1c = X0, X1

        self.mean_ = mu.detach().clone()          # keep the data's dtype and device

        Ur, A_tilde, Sr = self._build_operator(
            X0c.t(), X1c.t(), self.r, self.l2
        )
        self.basis_ = Ur.detach().clone()          # keep the data's dtype and device
        self.A_ = A_tilde.detach().clone()          # keep the data's dtype and device
        self._singular_values = Sr.clone()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def eig(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Eigendecomposition of the reduced operator A_tilde.

        Returns
        -------
        eigenvalues   : (r,) complex
        modes_reduced : (r, r) complex  — columns are reduced eigenvectors
        modes_full    : (D, r) complex  — full-space DMD modes (Phi = U W),
                        or None if basis_ is uninitialised.
        """
        if not self._fitted:
            raise RuntimeError("DMD not fitted.")
        lam, W = torch.linalg.eig(self.A_)
        Phi = self.basis_.to(W.dtype) @ W  # (D, r)
        return lam, W, Phi

    @property
    def explained_variance_ratio_(self) -> torch.Tensor | None:
        """Energy fraction captured by each singular value (after fit)."""
        if self._singular_values is None:
            return None
        s2 = self._singular_values ** 2
        return s2 / s2.sum().clamp_min(1e-12)

    def frequencies(self, dt: float = 1.0) -> torch.Tensor | None:
        """
        Continuous-time frequencies from DMD eigenvalues: f = imag(log(lam)) / (2pi*dt).
        Returns real tensor of shape (r,).  dt is the physical time step.
        """
        lam, _, _ = self.eig()
        log_lam = torch.log(lam + 1e-30)
        return (log_lam.imag / (2.0 * torch.pi * dt))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x0 : (B, D) — initial state in original space.
        Returns (B, steps+1, D).
        """
        if not self._fitted:
            raise RuntimeError("DMD not fitted.")
        if x0.ndim != 2:
            raise ValueError(f"x0 must be (B, D), got {tuple(x0.shape)}")

        B, D = x0.shape
        if self.mean_.ndim != 2 or self.mean_.shape[-1] != D:
            raise RuntimeError(
                f"Fitted mean has shape {tuple(self.mean_.shape)} but x0 has D={D}."
            )

        a = (x0 - self.mean_) @ self.basis_
        xs = [x0]
        for _ in range(int(steps)):
            a = a @ self.A_.t()
            xs.append(a @ self.basis_.t() + self.mean_)
        return torch.stack(xs, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        Xseq = self._as_btd(X)
        B, T, D = Xseq.shape
        yhat = self.rollout(Xseq[:, 0], steps=T - 1)

        losses: dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=Xseq.device)}
        if return_loss:
            losses["mse"]   = self.mse(yhat, Xseq)
            losses["total"] = losses["mse"]

        y_out = yhat if X.ndim == 3 else yhat[0]
        return ROMOutput(
            y=y_out,
            losses=losses,
            extras={
                "fitted": self._fitted,
                "rank":   int(self.basis_.shape[-1]) if self._fitted else 0,
            },
        )


class DMDc:
    """Dynamic Mode Decomposition with control (Proctor, Brunton & Kutz 2016), NumPy.

    Identifies the linear, input-driven model

        x_{k+1} = A x_k + B u_k (+ c)

    from snapshot pairs by least squares on the stacked data ``[X, U] W = Y`` (the formulation with unknown ``B`` of
    Proctor et al. 2016, solved without SVD truncation), optionally with a ridge penalty
    ``ridge * ||W||_F^2``. With ``affine=True`` a constant column is appended and a bias ``c`` is identified too
    (useful for states that are not centred, e.g. POD coefficients of a heated flow). Rows are samples:
    ``X`` is ``(N, n)``, ``U`` is ``(N, m)``. To work in a reduced space, encode the snapshots first (for example
    with :class:`~pinneapple_neural.architectures.rom.pod.POD`) and decode the rollout.

    Reference: J. L. Proctor, S. L. Brunton, J. N. Kutz (2016), Dynamic mode decomposition with control,
    *SIAM J. Applied Dynamical Systems* 15(1), 142-161.

    Attributes (after :meth:`fit`): ``A_`` (n, n), ``B_`` (n, m), ``c_`` (n,) (zeros unless ``affine``).
    """

    def __init__(self, affine: bool = False):
        self.affine = bool(affine)
        self.A_: np.ndarray | None = None
        self.B_: np.ndarray | None = None
        self.c_: np.ndarray | None = None

    def fit(self, X, U, Y=None, ridge: float = 0.0) -> DMDc:
        """Fit ``A``, ``B`` (and ``c``).

        Parameters
        ----------
        X : (T, n) one trajectory when ``Y`` is None (pairs ``X[k] -> X[k+1]``), else (N, n) current states.
        U : (T - 1, m) or (T, m) inputs of the trajectory (only the first ``T - 1`` are used), else (N, m).
        Y : (N, n) next states, or None.
        ridge : Tikhonov weight on all coefficients (including the bias); 0 gives plain least squares.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        U = np.asarray(U, dtype=np.float64)
        U = U.reshape(len(U), -1)
        if Y is None:
            if len(U) < len(X) - 1:
                raise ValueError(f"a trajectory of {len(X)} states needs at least {len(X) - 1} inputs, got {len(U)}")
            X, Y, U = X[:-1], X[1:], U[: len(X) - 1]
        else:
            Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        if not (len(X) == len(U) == len(Y)) or X.shape != Y.shape:
            raise ValueError(f"inconsistent shapes X {X.shape}, U {U.shape}, Y {Y.shape}")
        n, m = X.shape[1], U.shape[1]
        cols = [X, U] + ([np.ones((len(X), 1))] if self.affine else [])
        Z = np.concatenate(cols, axis=1)
        if ridge > 0:
            W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ Y)
        else:
            W = np.linalg.lstsq(Z, Y, rcond=None)[0]
        self.A_, self.B_ = W[:n].T.copy(), W[n:n + m].T.copy()
        self.c_ = W[n + m].copy() if self.affine else np.zeros(n)
        return self

    def _check(self):
        if self.A_ is None:
            raise RuntimeError("DMDc not fitted.")

    def predict(self, x, u) -> np.ndarray:
        """One step: ``x`` (n,) or (N, n), ``u`` (m,) or (N, m) -> next state(s), same leading shape as ``x``."""
        self._check()
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        u = u.reshape(x.shape[:-1] + (self.B_.shape[1],))
        return x @ self.A_.T + u @ self.B_.T + self.c_

    def rollout(self, x0, U) -> np.ndarray:
        """Free-run from ``x0`` (n,) under the input sequence ``U`` (K, m); returns (K + 1, n) starting with ``x0``."""
        self._check()
        x = np.asarray(x0, dtype=np.float64).reshape(-1)
        U = np.asarray(U, dtype=np.float64).reshape(len(U), -1)
        out = [x]
        for u in U:
            x = self.predict(x, u)
            out.append(x)
        return np.stack(out)
